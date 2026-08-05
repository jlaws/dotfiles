"""macOS ``defaults`` settings: the registry plus apply and guarded revert.

``SETTINGS`` is the single source of truth. The same list drives applying settings and reverting
them. Revert is guarded: a key is only restored if its current value still equals what setup
applied, so changes made after setup are left alone.
"""

from __future__ import annotations

import logging
import plistlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from macos_setup.archive import Archive
    from macos_setup.shell import Runner

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Setting:
    """One macOS default.

    ``type`` is ``bool``/``int``/``float``/``string`` for scalars, or ``raw`` when ``value`` is the
    literal argument list to pass after ``defaults write <domain> <key>`` (arrays, dicts, etc.).
    ``scope`` is ``user``, ``sudo``, or ``host`` (``defaults -currentHost``).
    """

    domain: str
    key: str
    type: str
    value: Any
    scope: str = "user"
    min_version: str | None = None


@dataclass
class MergeResult:
    """Outcome of :func:`merge_revert`: keys to restore, keys to delete, keys left alone.

    ``defaults import`` merges rather than replaces, so restores and deletes are separate: restored
    keys are re-imported with their original value, deleted keys are removed with ``defaults delete``.
    """

    restore: dict[str, Any] = field(default_factory=dict)
    delete: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def reverted(self) -> list[str]:
        """Keys that will change (restored or deleted)."""
        return list(self.restore) + self.delete


def build_write_args(setting: Setting) -> list[str]:
    """Return the argument list following ``defaults write <domain> <key>``."""
    if setting.type == "bool":
        return ["-bool", "true" if setting.value else "false"]
    if setting.type == "int":
        return ["-int", str(setting.value)]
    if setting.type == "float":
        return ["-float", str(setting.value)]
    if setting.type == "string":
        return ["-string", setting.value]
    if setting.type == "raw":
        return list(setting.value)
    raise ValueError(f"Unknown setting type: {setting.type!r}")


def guard_matches(current: Any, applied: Any) -> bool:
    """Return True when the live value still equals what setup applied."""
    return current == applied


def merge_revert(
    live: dict[str, Any],
    saved: dict[str, Any],
    keys: list[str],
    applied: dict[str, Any],
) -> MergeResult:
    """Build the domain dict to write back, reverting only keys setup still owns.

    For each key in ``keys``: if the live value still equals ``applied[key]`` (guard passes),
    restore it from ``saved`` (or delete it when it was absent before setup); otherwise leave the
    live value untouched and mark it skipped.
    """
    result = MergeResult()
    for key in keys:
        current = live.get(key)
        if key in live and guard_matches(current, applied.get(key)):
            if key in saved:
                result.restore[key] = saved[key]
            else:
                result.delete.append(key)
        else:
            result.skipped.append(key)
    return result


def version_gte(current: str, required: str) -> bool:
    """Return True when macOS version ``current`` is >= ``required`` (dotted numeric)."""
    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split("."))

    return parts(current) >= parts(required)


def applicable(settings: list[Setting], macos_version: str) -> list[Setting]:
    """Filter settings to those whose ``min_version`` is satisfied by ``macos_version``."""
    return [
        s
        for s in settings
        if s.min_version is None or version_gte(macos_version, s.min_version)
    ]


def domain_snapshot_name(domain: str) -> str:
    """Return a filesystem-safe filename for a domain's plist snapshot."""
    safe = domain.strip("/").replace("/", "_")
    return f"{safe}.plist"


@dataclass
class RevertSummary:
    """Aggregated (domain, key) pairs reverted or skipped across all domains."""

    reverted: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _defaults_argv(scope: str, rest: list[str]) -> list[str]:
    """Build a ``defaults`` argv, adding ``-currentHost`` for host scope."""
    argv = ["defaults"]
    if scope == "host":
        argv.append("-currentHost")
    return argv + rest


def export_domain(runner: Runner, scope: str, domain: str) -> dict[str, Any]:
    """Return a domain's current defaults as a native dict ({} when the domain is absent)."""
    result = runner.run(
        _defaults_argv(scope, ["export", domain, "-"]),
        sudo=(scope == "sudo"),
        capture=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    return plistlib.loads(result.stdout.encode())


def _group_by_domain(settings: list[Setting]) -> list[tuple[str, str, list[Setting]]]:
    """Group settings by (scope, domain), preserving first-seen order."""
    groups: dict[tuple[str, str], list[Setting]] = {}
    for setting in settings:
        groups.setdefault((setting.scope, setting.domain), []).append(setting)
    return [(scope, domain, items) for (scope, domain), items in groups.items()]


def apply_defaults(
    settings: list[Setting],
    archive: Archive,
    runner: Runner,
    macos_version: str,
    *,
    dry_run: bool = False,
) -> None:
    """Apply settings, snapshotting each domain first and recording originals + applied values."""
    for scope, domain, group in _group_by_domain(applicable(settings, macos_version)):
        if dry_run:
            for setting in group:
                _LOG.info(
                    "would set %s %s %s", domain, setting.key, " ".join(build_write_args(setting))
                )
            continue
        _LOG.info("Configuring %s (%d settings)", domain, len(group))
        before = export_domain(runner, scope, domain)
        snapshot = archive.domains_dir / domain_snapshot_name(domain)
        snapshot.write_bytes(plistlib.dumps(before))
        for setting in group:
            runner.run(
                _defaults_argv(scope, ["write", domain, setting.key, *build_write_args(setting)]),
                sudo=(scope == "sudo"),
            )
            _LOG.debug("set %s %s", domain, setting.key)
        after = export_domain(runner, scope, domain)
        for setting in group:
            archive.record_setting(
                scope,
                domain,
                setting.key,
                present=setting.key in before,
                applied=after.get(setting.key),
            )
    if not dry_run:
        archive.save()


def revert_defaults(archive: Archive, runner: Runner, *, dry_run: bool = False) -> RevertSummary:
    """Revert recorded settings, importing the merged domain only for keys setup still owns."""
    summary = RevertSummary()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in archive.manifest.get("settings", []):
        groups.setdefault((record["scope"], record["domain"]), []).append(record)

    for (scope, domain), records in groups.items():
        live = export_domain(runner, scope, domain)
        snapshot = archive.domains_dir / domain_snapshot_name(domain)
        saved = plistlib.loads(snapshot.read_bytes()) if snapshot.exists() else {}
        # A key may be written more than once (repeated -dict-add); revert it once.
        keys = list(dict.fromkeys(record["key"] for record in records))
        applied = {record["key"]: record["applied"] for record in records}
        result = merge_revert(live, saved, keys, applied)

        for key in result.reverted:
            _LOG.info("revert %s %s", domain, key)
            summary.reverted.append((domain, key))
        for key in result.skipped:
            _LOG.warning("skip (user-modified) %s %s", domain, key)
            summary.skipped.append((domain, key))

        if dry_run:
            continue
        # `defaults import` merges, so restore known keys via a small plist and delete the rest.
        if result.restore:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "restore.plist"
                path.write_bytes(plistlib.dumps(result.restore))
                runner.run(_defaults_argv(scope, ["import", domain, str(path)]), sudo=(scope == "sudo"))
        for key in result.delete:
            runner.run(_defaults_argv(scope, ["delete", domain, key]), sudo=(scope == "sudo"))
    return summary


_HOME = Path.home()
_LOGINWINDOW = "/Library/Preferences/com.apple.loginwindow"
_WINDOWSERVER = "/Library/Preferences/com.apple.windowserver"

# Spotlight search categories (web suggestions disabled for privacy).
_SPOTLIGHT_ITEMS = [
    '{"enabled" = 1;"name" = "APPLICATIONS";}',
    '{"enabled" = 1;"name" = "SYSTEM_PREFS";}',
    '{"enabled" = 1;"name" = "DIRECTORIES";}',
    '{"enabled" = 1;"name" = "PDF";}',
    '{"enabled" = 1;"name" = "DOCUMENTS";}',
    '{"enabled" = 0;"name" = "FONTS";}',
    '{"enabled" = 0;"name" = "MESSAGES";}',
    '{"enabled" = 0;"name" = "CONTACT";}',
    '{"enabled" = 0;"name" = "EVENT_TODO";}',
    '{"enabled" = 1;"name" = "IMAGES";}',
    '{"enabled" = 0;"name" = "BOOKMARKS";}',
    '{"enabled" = 0;"name" = "MUSIC";}',
    '{"enabled" = 0;"name" = "MOVIES";}',
    '{"enabled" = 1;"name" = "PRESENTATIONS";}',
    '{"enabled" = 1;"name" = "SPREADSHEETS";}',
    '{"enabled" = 1;"name" = "SOURCE";}',
    '{"enabled" = 0;"name" = "MENU_WEBSEARCH";}',
    '{"enabled" = 0;"name" = "MENU_SPOTLIGHT_SUGGESTIONS";}',
]

# The single source of truth for macOS `defaults`. Drives both apply and guarded revert.
# System-level commands (nvram/pmset/systemsetup/firewall/chflags) live in `system.py`.
SETTINGS: list[Setting] = [
    # --- General UI/UX ---
    Setting("com.apple.universalaccess", "reduceTransparency", "bool", True, scope="sudo"),
    Setting("NSGlobalDomain", "AppleHighlightColor", "string", "0.764700 0.976500 0.568600"),
    Setting("NSGlobalDomain", "NSTableViewDefaultSizeMode", "int", 2),
    Setting("NSGlobalDomain", "AppleShowScrollBars", "string", "Always"),
    Setting("NSGlobalDomain", "NSUseAnimatedFocusRing", "bool", False),
    Setting("NSGlobalDomain", "NSToolbarTitleViewRolloverDelay", "float", 0.0),
    Setting("NSGlobalDomain", "NSWindowResizeTime", "float", 0.001),
    Setting("NSGlobalDomain", "NSNavPanelExpandedStateForSaveMode", "bool", True),
    Setting("NSGlobalDomain", "NSNavPanelExpandedStateForSaveMode2", "bool", True),
    Setting("NSGlobalDomain", "PMPrintingExpandedStateForPrint", "bool", True),
    Setting("NSGlobalDomain", "PMPrintingExpandedStateForPrint2", "bool", True),
    Setting("NSGlobalDomain", "NSDocumentSaveNewDocumentsToCloud", "bool", False),
    Setting("com.apple.print.PrintingPrefs", "Quit When Finished", "bool", True),
    Setting("com.apple.LaunchServices", "LSQuarantine", "bool", False),
    Setting("NSGlobalDomain", "NSTextShowsControlCharacters", "bool", True),
    Setting("com.apple.systempreferences", "NSQuitAlwaysKeepsWindows", "bool", False),
    Setting("NSGlobalDomain", "NSDisableAutomaticTermination", "bool", True),
    Setting("com.apple.helpviewer", "DevMode", "bool", True),
    Setting(_LOGINWINDOW, "AdminHostInfo", "string", "HostName", scope="sudo"),
    Setting("NSGlobalDomain", "NSAutomaticCapitalizationEnabled", "bool", False),
    Setting("NSGlobalDomain", "NSAutomaticDashSubstitutionEnabled", "bool", False),
    Setting("NSGlobalDomain", "NSAutomaticPeriodSubstitutionEnabled", "bool", False),
    Setting("NSGlobalDomain", "NSAutomaticQuoteSubstitutionEnabled", "bool", False),
    Setting("NSGlobalDomain", "NSAutomaticSpellingCorrectionEnabled", "bool", False),
    # --- Input Devices ---
    Setting("com.apple.driver.AppleBluetoothMultitouch.trackpad", "Clicking", "bool", True),
    Setting("NSGlobalDomain", "com.apple.mouse.tapBehavior", "int", 1, scope="host"),
    Setting("NSGlobalDomain", "com.apple.mouse.tapBehavior", "int", 1),
    Setting(
        "com.apple.driver.AppleBluetoothMultitouch.trackpad",
        "TrackpadCornerSecondaryClick",
        "int",
        2,
    ),
    Setting("com.apple.driver.AppleBluetoothMultitouch.trackpad", "TrackpadRightClick", "bool", True),
    Setting(
        "NSGlobalDomain",
        "com.apple.trackpad.trackpadCornerClickBehavior",
        "int",
        1,
        scope="host",
    ),
    Setting("NSGlobalDomain", "com.apple.trackpad.enableSecondaryClick", "bool", True, scope="host"),
    Setting("NSGlobalDomain", "com.apple.swipescrolldirection", "bool", True),
    Setting("com.apple.BluetoothAudioAgent", "Apple Bitpool Min (editable)", "int", 40),
    Setting("NSGlobalDomain", "AppleKeyboardUIMode", "int", 3),
    Setting("com.apple.universalaccess", "closeViewScrollWheelToggle", "bool", True, scope="sudo"),
    Setting("com.apple.universalaccess", "HIDScrollZoomModifierMask", "int", 262144, scope="sudo"),
    Setting("com.apple.universalaccess", "closeViewZoomFollowsFocus", "bool", True, scope="sudo"),
    Setting("NSGlobalDomain", "ApplePressAndHoldEnabled", "bool", False),
    Setting("NSGlobalDomain", "KeyRepeat", "int", 5),
    Setting("NSGlobalDomain", "InitialKeyRepeat", "int", 25),
    Setting("NSGlobalDomain", "AppleLanguages", "raw", ["-array", "en"]),
    Setting("NSGlobalDomain", "AppleLocale", "string", "en_US@currency=USD"),
    Setting("NSGlobalDomain", "AppleMeasurementUnits", "string", "Inches"),
    Setting("NSGlobalDomain", "AppleMetricUnits", "bool", False),
    Setting(_LOGINWINDOW, "showInputMenu", "bool", False, scope="sudo"),
    # --- Screen ---
    Setting("com.apple.screensaver", "askForPassword", "int", 1),
    Setting("com.apple.screensaver", "askForPasswordDelay", "int", 0),
    Setting("com.apple.screencapture", "location", "string", str(_HOME / "Desktop")),
    Setting("com.apple.screencapture", "type", "string", "png"),
    Setting("com.apple.screencapture", "disable-shadow", "bool", True),
    Setting("NSGlobalDomain", "AppleFontSmoothing", "int", 1),
    Setting(_WINDOWSERVER, "DisplayResolutionEnabled", "bool", True, scope="sudo"),
    # --- Control Center & Menu Bar ---
    Setting("com.apple.menuextra.battery", "ShowPercent", "string", "YES"),
    Setting("com.apple.menuextra.clock", "ShowDate", "int", 1),
    Setting("com.apple.menuextra.clock", "ShowDayOfWeek", "bool", True),
    Setting("com.apple.menuextra.clock", "FlashDateSeparators", "bool", False),
    # --- Finder ---
    Setting("com.apple.finder", "QuitMenuItem", "bool", True),
    Setting("com.apple.finder", "DisableAllAnimations", "bool", True),
    Setting("com.apple.finder", "NewWindowTarget", "string", "PfDe"),
    Setting("com.apple.finder", "NewWindowTargetPath", "string", f"file://{_HOME}/Desktop/"),
    Setting("com.apple.finder", "ShowExternalHardDrivesOnDesktop", "bool", True),
    Setting("com.apple.finder", "ShowHardDrivesOnDesktop", "bool", True),
    Setting("com.apple.finder", "ShowMountedServersOnDesktop", "bool", True),
    Setting("com.apple.finder", "ShowRemovableMediaOnDesktop", "bool", True),
    Setting("com.apple.finder", "AppleShowAllFiles", "bool", True),
    Setting("NSGlobalDomain", "AppleShowAllExtensions", "bool", True),
    Setting("com.apple.finder", "ShowStatusBar", "bool", True),
    Setting("com.apple.finder", "ShowPathbar", "bool", True),
    Setting("com.apple.finder", "_FXShowPosixPathInTitle", "bool", True),
    Setting("com.apple.finder", "_FXSortFoldersFirst", "bool", True),
    Setting("com.apple.finder", "_FXSortFoldersFirstOnDesktop", "bool", True),
    Setting("com.apple.finder", "FXDefaultSearchScope", "string", "SCcf"),
    Setting("com.apple.finder", "FXEnableExtensionChangeWarning", "bool", False),
    Setting("NSGlobalDomain", "com.apple.springing.enabled", "bool", True),
    Setting("NSGlobalDomain", "com.apple.springing.delay", "float", 0.0),
    Setting("com.apple.desktopservices", "DSDontWriteNetworkStores", "bool", True),
    Setting("com.apple.desktopservices", "DSDontWriteUSBStores", "bool", True),
    Setting("com.apple.frameworks.diskimages", "skip-verify", "bool", True),
    Setting("com.apple.frameworks.diskimages", "skip-verify-locked", "bool", True),
    Setting("com.apple.frameworks.diskimages", "skip-verify-remote", "bool", True),
    Setting("com.apple.frameworks.diskimages", "auto-open-ro-root", "bool", True),
    Setting("com.apple.frameworks.diskimages", "auto-open-rw-root", "bool", True),
    Setting("com.apple.finder", "OpenWindowForNewRemovableDisk", "bool", True),
    Setting("com.apple.finder", "FXPreferredViewStyle", "string", "Nlsv"),
    Setting("com.apple.finder", "WarnOnEmptyTrash", "bool", False),
    Setting("com.apple.NetworkBrowser", "BrowseAllInterfaces", "bool", True),
    # AirDrop discoverability. `sharingd` expects the named mode the AirDrop pane
    # writes ("Off", "Contacts Only", "Everyone"), not an enum index.
    Setting("com.apple.sharingd", "DiscoverableMode", "string", "Contacts Only"),
    Setting(
        "com.apple.finder",
        "FXInfoPanesExpanded",
        "raw",
        [
            "-dict",
            "General", "-bool", "true",
            "OpenWith", "-bool", "true",
            "Privileges", "-bool", "true",
        ],
    ),
    # --- Spotlight ---
    Setting("com.apple.spotlight", "orderedItems", "raw", ["-array", *_SPOTLIGHT_ITEMS]),
    # --- Dock ---
    Setting("com.apple.dock", "mouse-over-hilite-stack", "bool", True),
    Setting("com.apple.dock", "tilesize", "int", 36),
    Setting("com.apple.dock", "mineffect", "string", "scale"),
    Setting("com.apple.dock", "minimize-to-application", "bool", True),
    Setting("com.apple.dock", "enable-spring-load-actions-on-all-items", "bool", True),
    Setting("com.apple.dock", "show-process-indicators", "bool", True),
    Setting("com.apple.dock", "launchanim", "bool", False),
    Setting("com.apple.dock", "expose-animation-duration", "float", 0.1),
    Setting("com.apple.dock", "expose-group-by-app", "bool", False),
    Setting("com.apple.dock", "mru-spaces", "bool", False),
    Setting("com.apple.dock", "autohide-delay", "float", 0.0),
    Setting("com.apple.dock", "autohide-time-modifier", "float", 0.0),
    Setting("com.apple.dock", "autohide", "bool", True),
    Setting("com.apple.dock", "showhidden", "bool", True),
    Setting("com.apple.dock", "show-recents", "bool", False),
    Setting("com.apple.dock", "wvous-tl-corner", "int", 1),
    Setting("com.apple.dock", "wvous-tl-modifier", "int", 1048576),
    Setting("com.apple.dock", "wvous-tr-corner", "int", 1),
    Setting("com.apple.dock", "wvous-tr-modifier", "int", 1048576),
    Setting("com.apple.dock", "wvous-bl-corner", "int", 1),
    Setting("com.apple.dock", "wvous-bl-modifier", "int", 1048576),
    Setting("com.apple.dock", "wvous-br-corner", "int", 1),
    Setting("com.apple.dock", "wvous-br-modifier", "int", 1048576),
    Setting("com.apple.dock", "contents-immutable", "bool", False),
    # --- Stage Manager & Window Management ---
    Setting("com.apple.WindowManager", "GloballyEnabled", "bool", False, min_version="13.0"),
    Setting("com.apple.WindowManager", "AutoHide", "bool", True, min_version="13.0"),
    Setting(
        "com.apple.WindowManager",
        "AppWindowGroupingBehavior",
        "bool",
        False,
        min_version="13.0",
    ),
    Setting(
        "com.apple.WindowManager",
        "EnableStandardClickToShowDesktop",
        "int",
        0,
        min_version="14.0",
    ),
    # --- Safari (sudo) ---
    Setting("com.apple.Safari", "UniversalSearchEnabled", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "SuppressSearchSuggestions", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "WebKitTabToLinksPreferenceKey", "bool", True, scope="sudo"),
    Setting(
        "com.apple.Safari",
        "com.apple.Safari.ContentPageGroupIdentifier.WebKit2TabsToLinks",
        "bool",
        True,
        scope="sudo",
    ),
    Setting("com.apple.Safari", "ShowFullURLInSmartSearchField", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "HomePage", "string", "about:blank", scope="sudo"),
    Setting("com.apple.Safari", "AutoOpenSafeDownloads", "bool", False, scope="sudo"),
    Setting(
        "com.apple.Safari",
        "com.apple.Safari.ContentPageGroupIdentifier.WebKit2BackspaceKeyNavigationEnabled",
        "bool",
        True,
        scope="sudo",
    ),
    Setting("com.apple.Safari", "ShowFavoritesBar", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "ShowSidebarInTopSites", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "DebugSnapshotsUpdatePolicy", "int", 2, scope="sudo"),
    Setting("com.apple.Safari", "IncludeInternalDebugMenu", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "FindOnPageMatchesWordStartsOnly", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "ProxiesInBookmarksBar", "raw", ["()"], scope="sudo"),
    Setting("com.apple.Safari", "IncludeDevelopMenu", "bool", True, scope="sudo"),
    Setting(
        "com.apple.Safari",
        "WebKitDeveloperExtrasEnabledPreferenceKey",
        "bool",
        True,
        scope="sudo",
    ),
    Setting(
        "com.apple.Safari",
        "com.apple.Safari.ContentPageGroupIdentifier.WebKit2DeveloperExtrasEnabled",
        "bool",
        True,
        scope="sudo",
    ),
    Setting("NSGlobalDomain", "WebKitDeveloperExtras", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "WebContinuousSpellCheckingEnabled", "bool", True, scope="sudo"),
    Setting(
        "com.apple.Safari", "WebAutomaticSpellingCorrectionEnabled", "bool", False, scope="sudo"
    ),
    Setting("com.apple.Safari", "AutoFillFromAddressBook", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "AutoFillPasswords", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "AutoFillCreditCardData", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "AutoFillMiscellaneousForms", "bool", False, scope="sudo"),
    Setting("com.apple.Safari", "WarnAboutFraudulentWebsites", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "WebKitPluginsEnabled", "bool", False, scope="sudo"),
    Setting(
        "com.apple.Safari",
        "com.apple.Safari.ContentPageGroupIdentifier.WebKit2PluginsEnabled",
        "bool",
        False,
        scope="sudo",
    ),
    Setting(
        "com.apple.Safari",
        "WebKitJavaScriptCanOpenWindowsAutomatically",
        "bool",
        False,
        scope="sudo",
    ),
    Setting(
        "com.apple.Safari",
        "com.apple.Safari.ContentPageGroupIdentifier.WebKit2JavaScriptCanOpenWindowsAutomatically",
        "bool",
        False,
        scope="sudo",
    ),
    Setting("com.apple.Safari", "SendDoNotTrackHTTPHeader", "bool", True, scope="sudo"),
    Setting("com.apple.Safari", "InstallExtensionUpdatesAutomatically", "bool", True, scope="sudo"),
    # --- Mail (sudo) ---
    Setting("com.apple.mail", "DisableReplyAnimations", "bool", True, scope="sudo"),
    Setting("com.apple.mail", "DisableSendAnimations", "bool", True, scope="sudo"),
    Setting("com.apple.mail", "AddressesIncludeNameOnPasteboard", "bool", False, scope="sudo"),
    Setting(
        "com.apple.mail",
        "NSUserKeyEquivalents",
        "raw",
        ["-dict-add", "Send", r"@\U21a9"],
        scope="sudo",
    ),
    Setting(
        "com.apple.mail",
        "DraftsViewerAttributes",
        "raw",
        ["-dict-add", "DisplayInThreadedMode", "-string", "yes"],
        scope="sudo",
    ),
    Setting(
        "com.apple.mail",
        "DraftsViewerAttributes",
        "raw",
        ["-dict-add", "SortedDescending", "-string", "yes"],
        scope="sudo",
    ),
    Setting(
        "com.apple.mail",
        "DraftsViewerAttributes",
        "raw",
        ["-dict-add", "SortOrder", "-string", "received-date"],
        scope="sudo",
    ),
    Setting("com.apple.mail", "DisableInlineAttachmentViewing", "bool", True, scope="sudo"),
    Setting(
        "com.apple.mail", "SpellCheckingBehavior", "string", "NoSpellCheckingEnabled", scope="sudo"
    ),
    # --- Terminal ---
    Setting("com.apple.terminal", "StringEncodings", "raw", ["-array", "4"]),
    Setting("com.apple.terminal", "SecureKeyboardEntry", "bool", True),
    Setting("com.apple.Terminal", "ShowLineMarks", "int", 0),
    # --- Activity Monitor ---
    Setting("com.apple.ActivityMonitor", "OpenMainWindow", "bool", True),
    Setting("com.apple.ActivityMonitor", "IconType", "int", 5),
    Setting("com.apple.ActivityMonitor", "ShowCategory", "int", 0),
    Setting("com.apple.ActivityMonitor", "SortColumn", "string", "CPUUsage"),
    Setting("com.apple.ActivityMonitor", "SortDirection", "int", 0),
    # --- TextEdit ---
    Setting("com.apple.TextEdit", "RichText", "int", 0),
    Setting("com.apple.TextEdit", "PlainTextEncoding", "int", 4),
    Setting("com.apple.TextEdit", "PlainTextEncodingForWrite", "int", 4),
    # --- Disk Utility ---
    Setting("com.apple.DiskUtility", "DUDebugMenuEnabled", "bool", True),
    Setting("com.apple.DiskUtility", "advanced-image-options", "bool", True),
    # --- Time Machine ---
    Setting("com.apple.TimeMachine", "DoNotOfferNewDisksForBackup", "bool", True),
    # --- Photos ---
    Setting("com.apple.ImageCapture", "disableHotPlug", "bool", True, scope="host"),
    # --- Messages ---
    Setting(
        "com.apple.messageshelper.MessageController",
        "SOInputLineSettings",
        "raw",
        ["-dict-add", "automaticEmojiSubstitutionEnablediMessage", "-bool", "false"],
    ),
    Setting(
        "com.apple.messageshelper.MessageController",
        "SOInputLineSettings",
        "raw",
        ["-dict-add", "automaticQuoteSubstitutionEnabled", "-bool", "false"],
    ),
    Setting(
        "com.apple.messageshelper.MessageController",
        "SOInputLineSettings",
        "raw",
        ["-dict-add", "continuousSpellCheckingEnabled", "-bool", "false"],
    ),
    # --- Chrome ---
    Setting("com.google.Chrome", "AppleEnableSwipeNavigateWithScrolls", "bool", False),
    Setting("com.google.Chrome", "AppleEnableMouseSwipeNavigateWithScrolls", "bool", False),
    Setting("com.google.Chrome", "DisablePrintPreview", "bool", False),
    Setting("com.google.Chrome", "PMPrintingExpandedStateForPrint2", "bool", True),
    # --- Mac App Store & Software Update ---
    Setting("com.apple.appstore", "WebKitDeveloperExtras", "bool", True),
    Setting("com.apple.appstore", "ShowDebugMenu", "bool", True),
    Setting("com.apple.SoftwareUpdate", "AutomaticCheckEnabled", "bool", True),
    Setting("com.apple.SoftwareUpdate", "ScheduleFrequency", "int", 1),
    Setting("com.apple.SoftwareUpdate", "AutomaticDownload", "int", 1),
    Setting("com.apple.SoftwareUpdate", "CriticalUpdateInstall", "int", 1),
    Setting("com.apple.SoftwareUpdate", "ConfigDataInstall", "int", 1),
    Setting("com.apple.commerce", "AutoUpdate", "bool", True),
    Setting("com.apple.commerce", "AutoUpdateRestartRequired", "bool", True),
]

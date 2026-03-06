#!/usr/bin/env -S uv run
"""macOS setup script: syncs dotfiles, installs packages, configures system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import NamedTuple

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Helpers
# =============================================================================


def run(
    *args: str,
    sudo: bool = False,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command, optionally with sudo."""
    cmd = ["sudo", *args] if sudo else list(args)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("#" * 79)
    print(f"# {title}")
    print("#" * 79)
    print()


def print_step(msg: str) -> None:
    """Print a step indicator."""
    print(f"  -> {msg}")


# =============================================================================
# Version detection
# =============================================================================


def get_macos_version() -> str:
    """Return the macOS product version string."""
    result = run("sw_vers", "-productVersion", capture=True)
    return result.stdout.strip()


def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon."""
    result = run("uname", "-m", capture=True)
    return result.stdout.strip() == "arm64"


def macos_version_gte(required: str) -> bool:
    """Return True if the current macOS version >= required."""
    current = get_macos_version()

    def _parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    return _parse(current) >= _parse(required)


# =============================================================================
# Pre-flight
# =============================================================================


def _sudo_keepalive() -> None:
    """Background thread to keep sudo credentials alive."""
    import time

    while True:
        subprocess.run(
            ["sudo", "-n", "true"],
            check=False,
            capture_output=True,
        )
        time.sleep(60)


def preflight_checks() -> None:
    """Close System Settings and request sudo."""
    print_section("Pre-flight Checks")

    print_step("Closing System Settings...")
    run(
        "osascript",
        "-e",
        'tell application "System Settings" to quit',
        check=False,
    )
    run(
        "osascript",
        "-e",
        'tell application "System Preferences" to quit',
        check=False,
    )

    print_step("Requesting administrator privileges...")
    run("sudo", "-v")

    # Keep sudo alive in background
    t = threading.Thread(target=_sudo_keepalive, daemon=True)
    t.start()


# =============================================================================
# Sync Dotfiles
# =============================================================================


def sync_dotfiles() -> None:
    """Pull latest and rsync dotfiles to home directory."""
    print_section("Syncing Dotfiles")

    print_step("Pulling latest changes from git...")
    run("git", "pull", "origin", "main")

    print_step("Copying dotfiles to home directory...")
    run(
        "rsync",
        "--exclude",
        ".git/",
        "--exclude",
        ".DS_Store",
        "--exclude",
        ".claude/",
        "--exclude",
        "setup.py",
        "--exclude",
        "setup.sh",
        "--exclude",
        "README.md",
        "--exclude",
        "CLAUDE.md",
        "--exclude",
        "ghosty_config.txt",
        "--exclude",
        "LICENSE",
        "--exclude",
        "pyproject.toml",
        "--exclude",
        ".python-version",
        "--exclude",
        "Makefile",
        "-avh",
        "--no-perms",
        str(SCRIPT_DIR) + "/",
        str(Path.home()) + "/",
    )


# =============================================================================
# Claude Code Configuration
# =============================================================================


def sync_claude(target: str = "") -> None:
    """Sync Claude Code configuration to target directory."""
    target_path = Path(target) if target else Path.home()

    print_section("Syncing Claude Code Configuration")
    print_step(f"Target: {target_path}/.claude/")

    claude_dir = target_path / ".claude"
    for subdir in ["commands", "skills", "agents", "references"]:
        (claude_dir / subdir).mkdir(parents=True, exist_ok=True)

    print_step(f"Syncing Claude configuration to {claude_dir}/...")
    run(
        "rsync",
        "-avh",
        "--no-perms",
        str(SCRIPT_DIR / ".claude" / "CLAUDE.md"),
        str(claude_dir / "CLAUDE.md"),
    )

    settings_file = claude_dir / "settings.json"
    if not settings_file.exists():
        print_step(f"Creating {settings_file}...")
        run(
            "rsync",
            "-avh",
            "--no-perms",
            str(SCRIPT_DIR / ".claude" / "settings.json"),
            str(settings_file),
        )
    else:
        print_step("Skipping settings.json (already exists)")

    print_step("Syncing Claude agents, commands, skills, and references...")
    for subdir in ["agents", "commands", "skills", "references"]:
        run(
            "rsync",
            "-avh",
            "--no-perms",
            str(SCRIPT_DIR / ".claude" / subdir) + "/",
            str(claude_dir / subdir) + "/",
        )


# =============================================================================
# Homebrew
# =============================================================================


def install_homebrew_packages() -> None:
    """Install Homebrew and packages."""
    print_section("Homebrew Setup")

    # Check if Homebrew is installed
    result = run("command", "-v", "brew", check=False, capture=True)
    if result.returncode != 0:
        print("Homebrew is required but not installed. Please install it first:")
        print(
            '  /bin/bash -c "$(curl -fsSL'
            " https://raw.githubusercontent.com/Homebrew/install/HEAD/"
            'install.sh)"'
        )
        sys.exit(1)

    print_step("Updating Homebrew...")
    run("brew", "update")

    print_step("Upgrading existing packages...")
    run("brew", "upgrade")

    # Save Homebrew prefix
    prefix_result = run("brew", "--prefix", capture=True)
    brew_prefix = prefix_result.stdout.strip()

    print_step("Installing GNU core utilities...")
    run("brew", "install", "coreutils")
    run(
        "ln",
        "-sf",
        f"{brew_prefix}/bin/gsha256sum",
        f"{brew_prefix}/bin/sha256sum",
        check=False,
    )

    print_step("Installing additional utilities...")
    run("brew", "install", "moreutils")
    run("brew", "install", "findutils")
    run("brew", "install", "gnu-sed")
    run("brew", "install", "wget")

    print_step("Installing updated macOS tools...")
    for pkg in ["vim", "grep", "openssh", "screen"]:
        run("brew", "install", pkg)

    print_step("Installing git tools...")
    for pkg in ["git", "git-lfs", "gh"]:
        run("brew", "install", pkg)

    print_step("Installing shell utilities...")
    run("brew", "install", "autojump")
    run("brew", "install", "mermaid-cli")

    print_step("Installing Rust toolchain...")
    run("brew", "install", "rustup")
    run("brew", "install", "mold")

    print_step("Installing Python toolchain...")
    run("brew", "install", "uv")

    print_step("Installing language servers...")
    run("brew", "install", "node")
    run("brew", "install", "pyright")
    run("brew", "install", "rust-analyzer")
    run(
        "npm",
        "install",
        "-g",
        "typescript-language-server",
        "typescript",
    )
    run("go", "install", "golang.org/x/tools/gopls@latest")

    print_step("Installing Claude Code...")
    run("bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash")

    print_step("Cleaning up...")
    run("brew", "cleanup")


# =============================================================================
# macOS Defaults (data-driven)
# =============================================================================


class Default(NamedTuple):
    """A macOS defaults write entry."""

    domain: str
    key: str
    type: str
    value: str
    sudo: bool = False
    current_host: bool = False


# fmt: off
GENERAL_UI: list[Default] = [
    Default("com.apple.universalaccess", "reduceTransparency", "-bool", "true", sudo=True),
    Default("NSGlobalDomain", "AppleHighlightColor", "-string", "0.764700 0.976500 0.568600"),
    Default("NSGlobalDomain", "NSTableViewDefaultSizeMode", "-int", "2"),
    Default("NSGlobalDomain", "AppleShowScrollBars", "-string", "Always"),
    Default("NSGlobalDomain", "NSUseAnimatedFocusRing", "-bool", "false"),
    Default("NSGlobalDomain", "NSToolbarTitleViewRolloverDelay", "-float", "0"),
    Default("NSGlobalDomain", "NSWindowResizeTime", "-float", "0.001"),
    Default("NSGlobalDomain", "NSNavPanelExpandedStateForSaveMode", "-bool", "true"),
    Default("NSGlobalDomain", "NSNavPanelExpandedStateForSaveMode2", "-bool", "true"),
    Default("NSGlobalDomain", "PMPrintingExpandedStateForPrint", "-bool", "true"),
    Default("NSGlobalDomain", "PMPrintingExpandedStateForPrint2", "-bool", "true"),
    Default("NSGlobalDomain", "NSDocumentSaveNewDocumentsToCloud", "-bool", "false"),
    Default("com.apple.print.PrintingPrefs", "Quit When Finished", "-bool", "true"),
    Default("com.apple.LaunchServices", "LSQuarantine", "-bool", "false"),
    Default("NSGlobalDomain", "NSTextShowsControlCharacters", "-bool", "true"),
    Default("com.apple.systempreferences", "NSQuitAlwaysKeepsWindows", "-bool", "false"),
    Default("NSGlobalDomain", "NSDisableAutomaticTermination", "-bool", "true"),
    Default("com.apple.helpviewer", "DevMode", "-bool", "true"),
    Default("/Library/Preferences/com.apple.loginwindow", "AdminHostInfo", "-string", "HostName", sudo=True),
    Default("NSGlobalDomain", "NSAutomaticCapitalizationEnabled", "-bool", "false"),
    Default("NSGlobalDomain", "NSAutomaticDashSubstitutionEnabled", "-bool", "false"),
    Default("NSGlobalDomain", "NSAutomaticPeriodSubstitutionEnabled", "-bool", "false"),
    Default("NSGlobalDomain", "NSAutomaticQuoteSubstitutionEnabled", "-bool", "false"),
    Default("NSGlobalDomain", "NSAutomaticSpellingCorrectionEnabled", "-bool", "false"),
]

INPUT_DEVICES: list[Default] = [
    Default("com.apple.driver.AppleBluetoothMultitouch.trackpad", "Clicking", "-bool", "true"),
    Default("NSGlobalDomain", "com.apple.mouse.tapBehavior", "-int", "1", current_host=True),
    Default("NSGlobalDomain", "com.apple.mouse.tapBehavior", "-int", "1"),
    Default("com.apple.driver.AppleBluetoothMultitouch.trackpad", "TrackpadCornerSecondaryClick", "-int", "2"),
    Default("com.apple.driver.AppleBluetoothMultitouch.trackpad", "TrackpadRightClick", "-bool", "true"),
    Default("NSGlobalDomain", "com.apple.trackpad.trackpadCornerClickBehavior", "-int", "1", current_host=True),
    Default("NSGlobalDomain", "com.apple.trackpad.enableSecondaryClick", "-bool", "true", current_host=True),
    Default("NSGlobalDomain", "com.apple.swipescrolldirection", "-bool", "true"),
    Default("com.apple.BluetoothAudioAgent", "Apple Bitpool Min (editable)", "-int", "40"),
    Default("NSGlobalDomain", "AppleKeyboardUIMode", "-int", "3"),
    Default("com.apple.universalaccess", "closeViewScrollWheelToggle", "-bool", "true", sudo=True),
    Default("com.apple.universalaccess", "HIDScrollZoomModifierMask", "-int", "262144", sudo=True),
    Default("com.apple.universalaccess", "closeViewZoomFollowsFocus", "-bool", "true", sudo=True),
    Default("NSGlobalDomain", "ApplePressAndHoldEnabled", "-bool", "false"),
    Default("NSGlobalDomain", "KeyRepeat", "-int", "5"),
    Default("NSGlobalDomain", "InitialKeyRepeat", "-int", "25"),
    Default("NSGlobalDomain", "AppleLanguages", "-array", "en"),
    Default("NSGlobalDomain", "AppleLocale", "-string", "en_US@currency=USD"),
    Default("NSGlobalDomain", "AppleMeasurementUnits", "-string", "Inches"),
    Default("NSGlobalDomain", "AppleMetricUnits", "-bool", "false"),
    Default("/Library/Preferences/com.apple.loginwindow", "showInputMenu", "-bool", "false", sudo=True),
]

SCREEN: list[Default] = [
    Default("com.apple.screensaver", "askForPassword", "-int", "1"),
    Default("com.apple.screensaver", "askForPasswordDelay", "-int", "0"),
    Default("com.apple.screencapture", "location", "-string", "${HOME}/Desktop"),
    Default("com.apple.screencapture", "type", "-string", "png"),
    Default("com.apple.screencapture", "disable-shadow", "-bool", "true"),
    Default("NSGlobalDomain", "AppleFontSmoothing", "-int", "1"),
    Default("/Library/Preferences/com.apple.windowserver", "DisplayResolutionEnabled", "-bool", "true", sudo=True),
]

CONTROL_CENTER: list[Default] = [
    Default("com.apple.menuextra.battery", "ShowPercent", "-string", "YES"),
    Default("com.apple.menuextra.clock", "ShowDate", "-int", "1"),
    Default("com.apple.menuextra.clock", "ShowDayOfWeek", "-bool", "true"),
    Default("com.apple.menuextra.clock", "FlashDateSeparators", "-bool", "false"),
]

FINDER: list[Default] = [
    Default("com.apple.finder", "QuitMenuItem", "-bool", "true"),
    Default("com.apple.finder", "DisableAllAnimations", "-bool", "true"),
    Default("com.apple.finder", "NewWindowTarget", "-string", "PfDe"),
    Default("com.apple.finder", "NewWindowTargetPath", "-string", "file://${HOME}/Desktop/"),
    Default("com.apple.finder", "ShowExternalHardDrivesOnDesktop", "-bool", "true"),
    Default("com.apple.finder", "ShowHardDrivesOnDesktop", "-bool", "true"),
    Default("com.apple.finder", "ShowMountedServersOnDesktop", "-bool", "true"),
    Default("com.apple.finder", "ShowRemovableMediaOnDesktop", "-bool", "true"),
    Default("com.apple.finder", "AppleShowAllFiles", "-bool", "true"),
    Default("NSGlobalDomain", "AppleShowAllExtensions", "-bool", "true"),
    Default("com.apple.finder", "ShowStatusBar", "-bool", "true"),
    Default("com.apple.finder", "ShowPathbar", "-bool", "true"),
    Default("com.apple.finder", "_FXShowPosixPathInTitle", "-bool", "true"),
    Default("com.apple.finder", "_FXSortFoldersFirst", "-bool", "true"),
    Default("com.apple.finder", "_FXSortFoldersFirstOnDesktop", "-bool", "true"),
    Default("com.apple.finder", "FXDefaultSearchScope", "-string", "SCcf"),
    Default("com.apple.finder", "FXEnableExtensionChangeWarning", "-bool", "false"),
    Default("NSGlobalDomain", "com.apple.springing.enabled", "-bool", "true"),
    Default("NSGlobalDomain", "com.apple.springing.delay", "-float", "0"),
    Default("com.apple.desktopservices", "DSDontWriteNetworkStores", "-bool", "true"),
    Default("com.apple.desktopservices", "DSDontWriteUSBStores", "-bool", "true"),
    Default("com.apple.frameworks.diskimages", "skip-verify", "-bool", "true"),
    Default("com.apple.frameworks.diskimages", "skip-verify-locked", "-bool", "true"),
    Default("com.apple.frameworks.diskimages", "skip-verify-remote", "-bool", "true"),
    Default("com.apple.frameworks.diskimages", "auto-open-ro-root", "-bool", "true"),
    Default("com.apple.frameworks.diskimages", "auto-open-rw-root", "-bool", "true"),
    Default("com.apple.finder", "OpenWindowForNewRemovableDisk", "-bool", "true"),
    Default("com.apple.finder", "FXPreferredViewStyle", "-string", "Nlsv"),
    Default("com.apple.finder", "WarnOnEmptyTrash", "-bool", "false"),
    Default("com.apple.NetworkBrowser", "BrowseAllInterfaces", "-bool", "true"),
    Default("com.apple.sharingd", "DiscoverableMode", "-int", "1"),
]

DOCK: list[Default] = [
    Default("com.apple.dock", "mouse-over-hilite-stack", "-bool", "true"),
    Default("com.apple.dock", "tilesize", "-int", "36"),
    Default("com.apple.dock", "mineffect", "-string", "scale"),
    Default("com.apple.dock", "minimize-to-application", "-bool", "true"),
    Default("com.apple.dock", "enable-spring-load-actions-on-all-items", "-bool", "true"),
    Default("com.apple.dock", "show-process-indicators", "-bool", "true"),
    Default("com.apple.dock", "launchanim", "-bool", "false"),
    Default("com.apple.dock", "expose-animation-duration", "-float", "0.1"),
    Default("com.apple.dock", "expose-group-by-app", "-bool", "false"),
    Default("com.apple.dock", "mru-spaces", "-bool", "false"),
    Default("com.apple.dock", "autohide-delay", "-float", "0"),
    Default("com.apple.dock", "autohide-time-modifier", "-float", "0"),
    Default("com.apple.dock", "autohide", "-bool", "true"),
    Default("com.apple.dock", "showhidden", "-bool", "true"),
    Default("com.apple.dock", "show-recents", "-bool", "false"),
    Default("com.apple.dock", "wvous-tl-corner", "-int", "1"),
    Default("com.apple.dock", "wvous-tl-modifier", "-int", "1048576"),
    Default("com.apple.dock", "wvous-tr-corner", "-int", "1"),
    Default("com.apple.dock", "wvous-tr-modifier", "-int", "1048576"),
    Default("com.apple.dock", "wvous-bl-corner", "-int", "1"),
    Default("com.apple.dock", "wvous-bl-modifier", "-int", "1048576"),
    Default("com.apple.dock", "wvous-br-corner", "-int", "1"),
    Default("com.apple.dock", "wvous-br-modifier", "-int", "1048576"),
    Default("com.apple.dock", "contents-immutable", "-bool", "false"),
]

STAGE_MANAGER: list[Default] = [
    Default("com.apple.WindowManager", "GloballyEnabled", "-bool", "false"),
    Default("com.apple.WindowManager", "AutoHide", "-bool", "true"),
    Default("com.apple.WindowManager", "AppWindowGroupingBehavior", "-bool", "false"),
]

STAGE_MANAGER_SONOMA: list[Default] = [
    Default("com.apple.WindowManager", "EnableStandardClickToShowDesktop", "-int", "0"),
]

SAFARI: list[Default] = [
    Default("com.apple.Safari", "UniversalSearchEnabled", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "SuppressSearchSuggestions", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "WebKitTabToLinksPreferenceKey", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "com.apple.Safari.ContentPageGroupIdentifier.WebKit2TabsToLinks", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "ShowFullURLInSmartSearchField", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "HomePage", "-string", "about:blank", sudo=True),
    Default("com.apple.Safari", "AutoOpenSafeDownloads", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "com.apple.Safari.ContentPageGroupIdentifier.WebKit2BackspaceKeyNavigationEnabled", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "ShowFavoritesBar", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "ShowSidebarInTopSites", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "DebugSnapshotsUpdatePolicy", "-int", "2", sudo=True),
    Default("com.apple.Safari", "IncludeInternalDebugMenu", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "FindOnPageMatchesWordStartsOnly", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "ProxiesInBookmarksBar", "-string", "()", sudo=True),
    Default("com.apple.Safari", "IncludeDevelopMenu", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "WebKitDeveloperExtrasEnabledPreferenceKey", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "com.apple.Safari.ContentPageGroupIdentifier.WebKit2DeveloperExtrasEnabled", "-bool", "true", sudo=True),
    Default("NSGlobalDomain", "WebKitDeveloperExtras", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "WebContinuousSpellCheckingEnabled", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "WebAutomaticSpellingCorrectionEnabled", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "AutoFillFromAddressBook", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "AutoFillPasswords", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "AutoFillCreditCardData", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "AutoFillMiscellaneousForms", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "WarnAboutFraudulentWebsites", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "WebKitPluginsEnabled", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "com.apple.Safari.ContentPageGroupIdentifier.WebKit2PluginsEnabled", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "WebKitJavaScriptCanOpenWindowsAutomatically", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "com.apple.Safari.ContentPageGroupIdentifier.WebKit2JavaScriptCanOpenWindowsAutomatically", "-bool", "false", sudo=True),
    Default("com.apple.Safari", "SendDoNotTrackHTTPHeader", "-bool", "true", sudo=True),
    Default("com.apple.Safari", "InstallExtensionUpdatesAutomatically", "-bool", "true", sudo=True),
]

MAIL: list[Default] = [
    Default("com.apple.mail", "DisableReplyAnimations", "-bool", "true", sudo=True),
    Default("com.apple.mail", "DisableSendAnimations", "-bool", "true", sudo=True),
    Default("com.apple.mail", "AddressesIncludeNameOnPasteboard", "-bool", "false", sudo=True),
    Default("com.apple.mail", "DisableInlineAttachmentViewing", "-bool", "true", sudo=True),
    Default("com.apple.mail", "SpellCheckingBehavior", "-string", "NoSpellCheckingEnabled", sudo=True),
]

TERMINAL: list[Default] = [
    Default("com.apple.terminal", "StringEncodings", "-array", "4"),
    Default("com.apple.terminal", "SecureKeyboardEntry", "-bool", "true"),
    Default("com.apple.Terminal", "ShowLineMarks", "-int", "0"),
]

ACTIVITY_MONITOR: list[Default] = [
    Default("com.apple.ActivityMonitor", "OpenMainWindow", "-bool", "true"),
    Default("com.apple.ActivityMonitor", "IconType", "-int", "5"),
    Default("com.apple.ActivityMonitor", "ShowCategory", "-int", "0"),
    Default("com.apple.ActivityMonitor", "SortColumn", "-string", "CPUUsage"),
    Default("com.apple.ActivityMonitor", "SortDirection", "-int", "0"),
]

TEXTEDIT: list[Default] = [
    Default("com.apple.TextEdit", "RichText", "-int", "0"),
    Default("com.apple.TextEdit", "PlainTextEncoding", "-int", "4"),
    Default("com.apple.TextEdit", "PlainTextEncodingForWrite", "-int", "4"),
]

OTHER_APPS: list[Default] = [
    Default("com.apple.DiskUtility", "DUDebugMenuEnabled", "-bool", "true"),
    Default("com.apple.DiskUtility", "advanced-image-options", "-bool", "true"),
    Default("com.apple.TimeMachine", "DoNotOfferNewDisksForBackup", "-bool", "true"),
    Default("com.apple.ImageCapture", "disableHotPlug", "-bool", "true", current_host=True),
    Default("com.google.Chrome", "AppleEnableSwipeNavigateWithScrolls", "-bool", "false"),
    Default("com.google.Chrome", "AppleEnableMouseSwipeNavigateWithScrolls", "-bool", "false"),
    Default("com.google.Chrome", "DisablePrintPreview", "-bool", "false"),
    Default("com.google.Chrome", "PMPrintingExpandedStateForPrint2", "-bool", "true"),
    Default("com.apple.appstore", "WebKitDeveloperExtras", "-bool", "true"),
    Default("com.apple.appstore", "ShowDebugMenu", "-bool", "true"),
    Default("com.apple.SoftwareUpdate", "AutomaticCheckEnabled", "-bool", "true"),
    Default("com.apple.SoftwareUpdate", "ScheduleFrequency", "-int", "1"),
    Default("com.apple.SoftwareUpdate", "AutomaticDownload", "-int", "1"),
    Default("com.apple.SoftwareUpdate", "CriticalUpdateInstall", "-int", "1"),
    Default("com.apple.SoftwareUpdate", "ConfigDataInstall", "-int", "1"),
    Default("com.apple.commerce", "AutoUpdate", "-bool", "true"),
    Default("com.apple.commerce", "AutoUpdateRestartRequired", "-bool", "true"),
]
# fmt: on


def apply_defaults(defaults: list[Default]) -> None:
    """Apply a list of macOS defaults write entries."""
    for d in defaults:
        cmd = ["defaults"]
        if d.current_host:
            cmd.append("-currentHost")
        cmd.extend(["write", d.domain, d.key, d.type, d.value])
        if d.sudo:
            cmd = ["sudo", *cmd]
        subprocess.run(cmd, check=True)


def configure_macos() -> None:
    """Configure macOS system preferences."""
    print_section("macOS System Preferences")

    # General UI/UX
    print_step("Configuring General UI/UX...")
    run("nvram", "SystemAudioVolume= ", sudo=True)
    apply_defaults(GENERAL_UI)

    # Input Devices
    print_step("Configuring input devices...")
    apply_defaults(INPUT_DEVICES)
    run("systemsetup", "-settimezone", "America/New_York", sudo=True)

    # Energy Settings
    print_step("Configuring energy settings...")
    energy_cmds = [
        ["pmset", "-a", "lidwake", "1"],
        ["pmset", "-a", "autorestart", "1"],
        ["systemsetup", "-setrestartfreeze", "on"],
        ["pmset", "-b", "displaysleep", "5"],
        ["pmset", "-c", "displaysleep", "15"],
        ["pmset", "-b", "sleep", "15"],
        ["pmset", "-c", "sleep", "0"],
        ["systemsetup", "-setcomputersleep", "Off"],
        ["pmset", "-b", "standbydelay", "3600"],
        ["pmset", "-c", "standbydelay", "86400"],
        ["pmset", "-a", "hibernatemode", "0"],
        ["pmset", "-b", "disksleep", "5"],
        ["pmset", "-c", "disksleep", "0"],
        ["pmset", "-b", "powernap", "0"],
        ["pmset", "-c", "powernap", "1"],
    ]
    for cmd in energy_cmds:
        run(*cmd, sudo=True)

    # Screen
    print_step("Configuring screen settings...")
    apply_defaults(SCREEN)

    # Control Center & Menu Bar
    print_step("Configuring Control Center & Menu Bar...")
    apply_defaults(CONTROL_CENTER)

    # Privacy & Security
    print_step("Configuring Privacy & Security...")
    firewall = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    run(firewall, "--setglobalstate", "on", sudo=True, check=False)
    run(firewall, "--setstealthmode", "on", sudo=True, check=False)
    run(
        "systemsetup",
        "-setwakeonnetworkaccess",
        "off",
        sudo=True,
        check=False,
    )

    # Finder
    print_step("Configuring Finder...")
    apply_defaults(FINDER)
    run("chflags", "nohidden", str(Path.home() / "Library"), check=False)
    run(
        "xattr",
        "-d",
        "com.apple.FinderInfo",
        str(Path.home() / "Library"),
        check=False,
    )
    run("chflags", "nohidden", "/Volumes", sudo=True)
    # Expanded file info panes
    run(
        "defaults",
        "write",
        "com.apple.finder",
        "FXInfoPanesExpanded",
        "-dict",
        "General",
        "-bool",
        "true",
        "OpenWith",
        "-bool",
        "true",
        "Privileges",
        "-bool",
        "true",
    )

    # Spotlight
    print_step("Configuring Spotlight...")
    spotlight_items = [
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
    run(
        "defaults",
        "write",
        "com.apple.spotlight",
        "orderedItems",
        "-array",
        *spotlight_items,
    )

    # Dock
    print_step("Configuring Dock...")
    apply_defaults(DOCK)

    # Stage Manager (macOS Ventura+)
    if macos_version_gte("13.0"):
        print_step("Configuring Stage Manager...")
        apply_defaults(STAGE_MANAGER)

    # Desktop click (macOS Sonoma+)
    if macos_version_gte("14.0"):
        apply_defaults(STAGE_MANAGER_SONOMA)

    # Safari
    print_step("Configuring Safari...")
    apply_defaults(SAFARI)

    # Mail
    print_step("Configuring Mail...")
    apply_defaults(MAIL)
    # Mail dict-add entries need special handling
    mail_dict_cmds = [
        [
            "sudo",
            "defaults",
            "write",
            "com.apple.mail",
            "NSUserKeyEquivalents",
            "-dict-add",
            "Send",
            "@\u21a9",
        ],
        [
            "sudo",
            "defaults",
            "write",
            "com.apple.mail",
            "DraftsViewerAttributes",
            "-dict-add",
            "DisplayInThreadedMode",
            "-string",
            "yes",
        ],
        [
            "sudo",
            "defaults",
            "write",
            "com.apple.mail",
            "DraftsViewerAttributes",
            "-dict-add",
            "SortedDescending",
            "-string",
            "yes",
        ],
        [
            "sudo",
            "defaults",
            "write",
            "com.apple.mail",
            "DraftsViewerAttributes",
            "-dict-add",
            "SortOrder",
            "-string",
            "received-date",
        ],
    ]
    for cmd in mail_dict_cmds:
        subprocess.run(cmd, check=True)

    # Terminal
    print_step("Configuring Terminal...")
    apply_defaults(TERMINAL)

    # Other Apps
    print_step("Configuring other applications...")
    apply_defaults(ACTIVITY_MONITOR)
    apply_defaults(TEXTEDIT)
    apply_defaults(OTHER_APPS)
    # Messages dict-add entries
    messages_dict_cmds = [
        [
            "defaults",
            "write",
            "com.apple.messageshelper.MessageController",
            "SOInputLineSettings",
            "-dict-add",
            "automaticEmojiSubstitutionEnablediMessage",
            "-bool",
            "false",
        ],
        [
            "defaults",
            "write",
            "com.apple.messageshelper.MessageController",
            "SOInputLineSettings",
            "-dict-add",
            "automaticQuoteSubstitutionEnabled",
            "-bool",
            "false",
        ],
        [
            "defaults",
            "write",
            "com.apple.messageshelper.MessageController",
            "SOInputLineSettings",
            "-dict-add",
            "continuousSpellCheckingEnabled",
            "-bool",
            "false",
        ],
    ]
    for cmd in messages_dict_cmds:
        subprocess.run(cmd, check=True)


# =============================================================================
# Restart Apps
# =============================================================================


RESTART_APPS = [
    "Activity Monitor",
    "cfprefsd",
    "Dock",
    "Finder",
    "Google Chrome",
    "Mail",
    "Messages",
    "Photos",
    "Safari",
    "SystemUIServer",
    "Terminal",
]


def restart_apps() -> None:
    """Restart affected applications."""
    print_section("Restarting Affected Applications")
    for app in RESTART_APPS:
        run("killall", app, check=False)


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="macOS setup script",
        add_help=False,
    )
    parser.add_argument(
        "-d",
        dest="dotfiles",
        action="store_true",
        help="Sync dotfiles to home directory",
    )
    parser.add_argument(
        "-c",
        dest="claude",
        action="store_true",
        help="Sync Claude Code configuration",
    )
    parser.add_argument(
        "-b",
        dest="brew",
        action="store_true",
        help="Install Homebrew packages",
    )
    parser.add_argument(
        "-m",
        dest="macos",
        action="store_true",
        help="Configure macOS system preferences",
    )
    parser.add_argument(
        "-r",
        dest="restart",
        action="store_true",
        help="Restart affected applications",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project",
        default="",
        help="Target project folder for Claude sync (requires -c)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show this help message",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Entry point."""
    print()
    print("===========================================================================")
    print("                           macOS Setup Script")
    print("===========================================================================")

    os.chdir(SCRIPT_DIR)
    args = parse_args()

    if args.show_help:
        print("Usage: ./setup.py [flags]")
        print()
        print("Flags (can be combined):")
        print("  -d    Sync dotfiles to home directory")
        print("  -c    Sync Claude Code configuration")
        print("  -b    Install Homebrew packages")
        print("  -m    Configure macOS system preferences")
        print("  -r    Restart affected applications")
        print("  -p <path>  Target project folder for Claude sync")
        print("  -f    Skip confirmation prompt")
        print("  -h    Show this help message")
        return

    # Validate project arg
    if args.project and not args.claude:
        print("Error: -p/--project requires -c")
        sys.exit(1)
    if args.project and not Path(args.project).is_dir():
        print(f"Error: Project directory does not exist: {args.project}")
        sys.exit(1)

    has_selection = any(
        [
            args.dotfiles,
            args.claude,
            args.brew,
            args.macos,
            args.restart,
        ]
    )

    # Pre-flight for steps that need sudo
    needs_sudo = not has_selection or args.brew or args.macos or args.restart
    if needs_sudo:
        preflight_checks()

    # No specific steps: run all
    if not has_selection:
        if not args.force:
            reply = input("This will overwrite files and change system settings. Continue? (y/n) ")
            if not reply.lower().startswith("y"):
                print("Aborted.")
                sys.exit(1)

        sync_dotfiles()
        sync_claude(args.project)
        install_homebrew_packages()
        configure_macos()
        restart_apps()

        print_section("Setup Complete")
        print("Some changes require a logout or restart to take effect.")
        print()
        print("Re-source your shell: `source ~/.zshrc`")
        return

    # Execute selected steps
    if args.dotfiles:
        sync_dotfiles()
    if args.claude:
        sync_claude(args.project)
    if args.brew:
        install_homebrew_packages()
    if args.macos:
        configure_macos()
    if args.restart:
        restart_apps()


if __name__ == "__main__":
    main()

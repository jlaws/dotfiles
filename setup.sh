#!/usr/bin/env zsh

# =============================================================================
# macOS Setup Script
# =============================================================================
# Consolidates dotfiles sync, Homebrew installation, and macOS configuration.
# Run with --force or -f to skip confirmation prompts.
#
# Usage:
#   ./setup.sh           # Interactive mode (prompts for confirmation)
#   ./setup.sh --force   # Skip all prompts
#   ./setup.sh -f        # Skip all prompts
# =============================================================================

set -e

SCRIPT_DIR="$(dirname "${0}")"
cd "$SCRIPT_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

print_section() {
	echo ""
	echo "###############################################################################"
	echo "# $1"
	echo "###############################################################################"
	echo ""
}

print_step() {
	echo "  → $1"
}

# macOS version detection for conditional settings
get_macos_major_version() {
	sw_vers -productVersion | cut -d '.' -f 1
}

is_apple_silicon() {
	[[ $(uname -m) == "arm64" ]]
}

macos_version_gte() {
	local required="$1"
	local current=$(sw_vers -productVersion)
	[[ "$(printf '%s\n' "$required" "$current" | sort -V | head -n1)" == "$required" ]]
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_section "Pre-flight Checks"

# Close System Settings/Preferences to prevent conflicts
print_step "Closing System Settings..."
osascript -e 'tell application "System Settings" to quit' 2>/dev/null || true
osascript -e 'tell application "System Preferences" to quit' 2>/dev/null || true

# Request sudo upfront
print_step "Requesting administrator privileges..."
sudo -v

# Keep sudo alive throughout the script
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# =============================================================================
# Sync Dotfiles
# =============================================================================

sync_dotfiles() {
	print_section "Syncing Dotfiles"

	print_step "Pulling latest changes from git..."
	git pull origin main

	print_step "Copying dotfiles to home directory..."
	rsync --exclude ".git/" \
		--exclude ".DS_Store" \
		--exclude "setup.sh" \
		--exclude "README.md" \
		--exclude "LICENSE-MIT.txt" \
		--exclude "init/" \
		-avh --no-perms . ~
}

# =============================================================================
# Homebrew Setup
# =============================================================================

install_homebrew_packages() {
	print_section "Homebrew Setup"

	# Check if Homebrew is installed
	if ! command -v brew &> /dev/null; then
		print_step "Installing Homebrew..."
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	fi

	print_step "Updating Homebrew..."
	brew update

	print_step "Upgrading existing packages..."
	brew upgrade

	# Save Homebrew's installed location
	BREW_PREFIX=$(brew --prefix)

	print_step "Installing GNU core utilities..."
	brew install coreutils
	ln -sf "${BREW_PREFIX}/bin/gsha256sum" "${BREW_PREFIX}/bin/sha256sum" 2>/dev/null || true

	print_step "Installing additional utilities..."
	brew install moreutils      # Useful utilities like sponge
	brew install findutils      # GNU find, locate, updatedb, xargs
	brew install gnu-sed        # GNU sed
	brew install wget           # wget with IRI support

	print_step "Installing updated macOS tools..."
	brew install vim
	brew install grep
	brew install openssh
	brew install screen

	print_step "Installing git tools..."
	brew install git
	brew install git-lfs

	print_step "Installing shell utilities..."
	brew install autojump

	print_step "Cleaning up..."
	brew cleanup
}

# =============================================================================
# macOS System Preferences
# =============================================================================

configure_macos() {
	print_section "macOS System Preferences"

	# -------------------------------------------------------------------------
	# General UI/UX
	# -------------------------------------------------------------------------
	print_step "Configuring General UI/UX..."

	# Disable boot sound
	sudo nvram SystemAudioVolume=" "

	# Reduce transparency
	sudo defaults write com.apple.universalaccess reduceTransparency -bool true

	# Set highlight color to green
	defaults write NSGlobalDomain AppleHighlightColor -string "0.764700 0.976500 0.568600"

	# Set sidebar icon size to medium
	defaults write NSGlobalDomain NSTableViewDefaultSizeMode -int 2

	# Always show scrollbars
	defaults write NSGlobalDomain AppleShowScrollBars -string "Always"

	# Disable focus ring animation
	defaults write NSGlobalDomain NSUseAnimatedFocusRing -bool false

	# Faster toolbar title rollover
	defaults write NSGlobalDomain NSToolbarTitleViewRolloverDelay -float 0

	# Faster window resize
	defaults write NSGlobalDomain NSWindowResizeTime -float 0.001

	# Expand save panel by default
	defaults write NSGlobalDomain NSNavPanelExpandedStateForSaveMode -bool true
	defaults write NSGlobalDomain NSNavPanelExpandedStateForSaveMode2 -bool true

	# Expand print panel by default
	defaults write NSGlobalDomain PMPrintingExpandedStateForPrint -bool true
	defaults write NSGlobalDomain PMPrintingExpandedStateForPrint2 -bool true

	# Save to disk by default (not iCloud)
	defaults write NSGlobalDomain NSDocumentSaveNewDocumentsToCloud -bool false

	# Auto-quit printer app when done
	defaults write com.apple.print.PrintingPrefs "Quit When Finished" -bool true

	# Disable app quarantine dialog
	defaults write com.apple.LaunchServices LSQuarantine -bool false

	# Show control characters
	defaults write NSGlobalDomain NSTextShowsControlCharacters -bool true

	# Disable Resume system-wide
	defaults write com.apple.systempreferences NSQuitAlwaysKeepsWindows -bool false

	# Disable automatic app termination
	defaults write NSGlobalDomain NSDisableAutomaticTermination -bool true

	# Non-floating Help Viewer
	defaults write com.apple.helpviewer DevMode -bool true

	# Show IP in login window
	sudo defaults write /Library/Preferences/com.apple.loginwindow AdminHostInfo HostName

	# Disable auto-capitalization
	defaults write NSGlobalDomain NSAutomaticCapitalizationEnabled -bool false

	# Disable smart dashes
	defaults write NSGlobalDomain NSAutomaticDashSubstitutionEnabled -bool false

	# Disable auto-period
	defaults write NSGlobalDomain NSAutomaticPeriodSubstitutionEnabled -bool false

	# Disable smart quotes
	defaults write NSGlobalDomain NSAutomaticQuoteSubstitutionEnabled -bool false

	# Disable auto-correct
	defaults write NSGlobalDomain NSAutomaticSpellingCorrectionEnabled -bool false

	# -------------------------------------------------------------------------
	# Input Devices
	# -------------------------------------------------------------------------
	print_step "Configuring input devices..."

	# Trackpad: tap to click
	defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad Clicking -bool true
	defaults -currentHost write NSGlobalDomain com.apple.mouse.tapBehavior -int 1
	defaults write NSGlobalDomain com.apple.mouse.tapBehavior -int 1

	# Trackpad: right-click in bottom right corner
	defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad TrackpadCornerSecondaryClick -int 2
	defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad TrackpadRightClick -bool true
	defaults -currentHost write NSGlobalDomain com.apple.trackpad.trackpadCornerClickBehavior -int 1
	defaults -currentHost write NSGlobalDomain com.apple.trackpad.enableSecondaryClick -bool true

	# Use natural scrolling
	defaults write NSGlobalDomain com.apple.swipescrolldirection -bool true

	# Better Bluetooth audio
	defaults write com.apple.BluetoothAudioAgent "Apple Bitpool Min (editable)" -int 40

	# Full keyboard access
	defaults write NSGlobalDomain AppleKeyboardUIMode -int 3

	# Ctrl+scroll to zoom
	sudo defaults write com.apple.universalaccess closeViewScrollWheelToggle -bool true
	sudo defaults write com.apple.universalaccess HIDScrollZoomModifierMask -int 262144
	sudo defaults write com.apple.universalaccess closeViewZoomFollowsFocus -bool true

	# Disable press-and-hold (enable key repeat)
	defaults write NSGlobalDomain ApplePressAndHoldEnabled -bool false

	# Fast keyboard repeat (values: lower = faster; 2/15 balances speed vs accidental repeats)
	defaults write NSGlobalDomain KeyRepeat -int 2
	defaults write NSGlobalDomain InitialKeyRepeat -int 15

	# Language and locale
	defaults write NSGlobalDomain AppleLanguages -array "en"
	defaults write NSGlobalDomain AppleLocale -string "en_US@currency=USD"
	defaults write NSGlobalDomain AppleMeasurementUnits -string "Inches"
	defaults write NSGlobalDomain AppleMetricUnits -bool false

	# Don't show language menu at login
	sudo defaults write /Library/Preferences/com.apple.loginwindow showInputMenu -bool false

	# Set timezone
	sudo systemsetup -settimezone "America/New_York" > /dev/null

	# -------------------------------------------------------------------------
	# Energy Settings
	# -------------------------------------------------------------------------
	print_step "Configuring energy settings..."

	sudo pmset -a lidwake 1
	sudo pmset -a autorestart 1
	sudo systemsetup -setrestartfreeze on

	# Display sleep: faster on battery
	sudo pmset -b displaysleep 5
	sudo pmset -c displaysleep 15

	# System sleep
	sudo pmset -b sleep 15
	sudo pmset -c sleep 0
	sudo systemsetup -setcomputersleep Off > /dev/null

	# Standby: enter sooner on battery (1 hour vs 24 hours)
	sudo pmset -b standbydelay 3600
	sudo pmset -c standbydelay 86400

	# Hibernate mode 0 = RAM only (no disk backup)
	sudo pmset -a hibernatemode 0

	# Disk sleep on battery
	sudo pmset -b disksleep 5
	sudo pmset -c disksleep 0

	# Disable Power Nap on battery (prevents wake for background tasks)
	sudo pmset -b powernap 0
	sudo pmset -c powernap 1

	# -------------------------------------------------------------------------
	# Screen
	# -------------------------------------------------------------------------
	print_step "Configuring screen settings..."

	# Password after sleep
	defaults write com.apple.screensaver askForPassword -int 1
	defaults write com.apple.screensaver askForPasswordDelay -int 0

	# Screenshots
	defaults write com.apple.screencapture location -string "${HOME}/Desktop"
	defaults write com.apple.screencapture type -string "png"
	defaults write com.apple.screencapture disable-shadow -bool true

	# Font smoothing
	defaults write NSGlobalDomain AppleFontSmoothing -int 1

	# HiDPI support
	sudo defaults write /Library/Preferences/com.apple.windowserver DisplayResolutionEnabled -bool true

	# -------------------------------------------------------------------------
	# Control Center & Menu Bar
	# -------------------------------------------------------------------------
	print_step "Configuring Control Center & Menu Bar..."

	# Show battery percentage in menu bar
	defaults write com.apple.menuextra.battery ShowPercent -string "YES"

	# Clock: show date in menu bar
	defaults write com.apple.menuextra.clock ShowDate -int 1

	# Clock: show day of week
	defaults write com.apple.menuextra.clock ShowDayOfWeek -bool true

	# Clock: flash time separators (disabled)
	defaults write com.apple.menuextra.clock FlashDateSeparators -bool false

	# -------------------------------------------------------------------------
	# Privacy & Security
	# -------------------------------------------------------------------------
	print_step "Configuring Privacy & Security..."

	# Enable firewall
	sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on 2>/dev/null || true

	# Enable firewall stealth mode (don't respond to ping)
	sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode on 2>/dev/null || true

	# Disable wake-on-network
	sudo systemsetup -setwakeonnetworkaccess off 2>/dev/null || true

	# -------------------------------------------------------------------------
	# Finder
	# -------------------------------------------------------------------------
	print_step "Configuring Finder..."

	# Allow quitting
	defaults write com.apple.finder QuitMenuItem -bool true

	# Disable animations
	defaults write com.apple.finder DisableAllAnimations -bool true

	# Default to Desktop
	defaults write com.apple.finder NewWindowTarget -string "PfDe"
	defaults write com.apple.finder NewWindowTargetPath -string "file://${HOME}/Desktop/"

	# Show all drives on desktop
	defaults write com.apple.finder ShowExternalHardDrivesOnDesktop -bool true
	defaults write com.apple.finder ShowHardDrivesOnDesktop -bool true
	defaults write com.apple.finder ShowMountedServersOnDesktop -bool true
	defaults write com.apple.finder ShowRemovableMediaOnDesktop -bool true

	# Show hidden files
	defaults write com.apple.finder AppleShowAllFiles -bool true

	# Show all extensions
	defaults write NSGlobalDomain AppleShowAllExtensions -bool true

	# Show status and path bars
	defaults write com.apple.finder ShowStatusBar -bool true
	defaults write com.apple.finder ShowPathbar -bool true

	# Full path in title
	defaults write com.apple.finder _FXShowPosixPathInTitle -bool true

	# Folders on top when sorting by name
	defaults write com.apple.finder _FXSortFoldersFirst -bool true

	# Keep folders on top on Desktop
	defaults write com.apple.finder _FXSortFoldersFirstOnDesktop -bool true

	# Search current folder
	defaults write com.apple.finder FXDefaultSearchScope -string "SCcf"

	# No extension change warning
	defaults write com.apple.finder FXEnableExtensionChangeWarning -bool false

	# Spring loading
	defaults write NSGlobalDomain com.apple.springing.enabled -bool true
	defaults write NSGlobalDomain com.apple.springing.delay -float 0

	# No .DS_Store on network/USB
	defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
	defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true

	# Skip disk verification
	defaults write com.apple.frameworks.diskimages skip-verify -bool true
	defaults write com.apple.frameworks.diskimages skip-verify-locked -bool true
	defaults write com.apple.frameworks.diskimages skip-verify-remote -bool true

	# Auto-open on mount
	defaults write com.apple.frameworks.diskimages auto-open-ro-root -bool true
	defaults write com.apple.frameworks.diskimages auto-open-rw-root -bool true
	defaults write com.apple.finder OpenWindowForNewRemovableDisk -bool true

	# List view by default
	defaults write com.apple.finder FXPreferredViewStyle -string "Nlsv"

	# No trash warning
	defaults write com.apple.finder WarnOnEmptyTrash -bool false

	# AirDrop over Ethernet and older Macs
	defaults write com.apple.NetworkBrowser BrowseAllInterfaces -bool true

	# AirDrop: Contacts only for security (0=off, 1=contacts, 2=everyone)
	defaults write com.apple.sharingd DiscoverableMode -int 1

	# Show ~/Library
	chflags nohidden ~/Library 2>/dev/null || true
	xattr -d com.apple.FinderInfo ~/Library 2>/dev/null || true

	# Show /Volumes
	sudo chflags nohidden /Volumes

	# Expanded file info panes
	defaults write com.apple.finder FXInfoPanesExpanded -dict \
		General -bool true \
		OpenWith -bool true \
		Privileges -bool true

	# -------------------------------------------------------------------------
	# Spotlight
	# -------------------------------------------------------------------------
	print_step "Configuring Spotlight..."

	# Configure Spotlight search categories (disable web suggestions for privacy)
	defaults write com.apple.spotlight orderedItems -array \
		'{"enabled" = 1;"name" = "APPLICATIONS";}' \
		'{"enabled" = 1;"name" = "SYSTEM_PREFS";}' \
		'{"enabled" = 1;"name" = "DIRECTORIES";}' \
		'{"enabled" = 1;"name" = "PDF";}' \
		'{"enabled" = 1;"name" = "DOCUMENTS";}' \
		'{"enabled" = 0;"name" = "FONTS";}' \
		'{"enabled" = 0;"name" = "MESSAGES";}' \
		'{"enabled" = 0;"name" = "CONTACT";}' \
		'{"enabled" = 0;"name" = "EVENT_TODO";}' \
		'{"enabled" = 1;"name" = "IMAGES";}' \
		'{"enabled" = 0;"name" = "BOOKMARKS";}' \
		'{"enabled" = 0;"name" = "MUSIC";}' \
		'{"enabled" = 0;"name" = "MOVIES";}' \
		'{"enabled" = 1;"name" = "PRESENTATIONS";}' \
		'{"enabled" = 1;"name" = "SPREADSHEETS";}' \
		'{"enabled" = 1;"name" = "SOURCE";}' \
		'{"enabled" = 0;"name" = "MENU_WEBSEARCH";}' \
		'{"enabled" = 0;"name" = "MENU_SPOTLIGHT_SUGGESTIONS";}'

	# -------------------------------------------------------------------------
	# Dock
	# -------------------------------------------------------------------------
	print_step "Configuring Dock..."

	defaults write com.apple.dock mouse-over-hilite-stack -bool true
	defaults write com.apple.dock tilesize -int 36
	defaults write com.apple.dock mineffect -string "scale"
	defaults write com.apple.dock minimize-to-application -bool true
	defaults write com.apple.dock enable-spring-load-actions-on-all-items -bool true
	defaults write com.apple.dock show-process-indicators -bool true
	defaults write com.apple.dock launchanim -bool false
	defaults write com.apple.dock expose-animation-duration -float 0.1
	defaults write com.apple.dock expose-group-by-app -bool false
	defaults write com.apple.dock mru-spaces -bool false
	defaults write com.apple.dock autohide-delay -float 0
	defaults write com.apple.dock autohide-time-modifier -float 0
	defaults write com.apple.dock autohide -bool true
	defaults write com.apple.dock showhidden -bool true
	defaults write com.apple.dock show-recents -bool false

	# Hot corners: all disabled
	defaults write com.apple.dock wvous-tl-corner -int 1
	defaults write com.apple.dock wvous-tl-modifier -int 1048576
	defaults write com.apple.dock wvous-tr-corner -int 1
	defaults write com.apple.dock wvous-tr-modifier -int 1048576
	defaults write com.apple.dock wvous-bl-corner -int 1
	defaults write com.apple.dock wvous-bl-modifier -int 1048576
	defaults write com.apple.dock wvous-br-corner -int 1
	defaults write com.apple.dock wvous-br-modifier -int 1048576

	# Lock Dock contents to prevent accidental removal
	defaults write com.apple.dock contents-immutable -bool true

	# -------------------------------------------------------------------------
	# Stage Manager & Window Management (macOS Ventura+)
	# -------------------------------------------------------------------------
	if macos_version_gte "13.0"; then
		print_step "Configuring Stage Manager..."

		# Disable Stage Manager (set to true to enable)
		defaults write com.apple.WindowManager GloballyEnabled -bool false

		# Auto-hide recent apps strip when using Stage Manager
		defaults write com.apple.WindowManager AutoHide -bool true

		# Group windows by application
		defaults write com.apple.WindowManager AppWindowGroupingBehavior -bool false
	fi

	# Desktop click behavior (macOS Sonoma+)
	if macos_version_gte "14.0"; then
		# Click wallpaper to reveal desktop: 0=Only in Stage Manager, 1=Always
		defaults write com.apple.WindowManager EnableStandardClickToShowDesktop -int 0
	fi

	# -------------------------------------------------------------------------
	# Safari
	# -------------------------------------------------------------------------
	print_step "Configuring Safari..."

	sudo defaults write com.apple.Safari UniversalSearchEnabled -bool false
	sudo defaults write com.apple.Safari SuppressSearchSuggestions -bool true
	sudo defaults write com.apple.Safari WebKitTabToLinksPreferenceKey -bool true
	sudo defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2TabsToLinks -bool true
	sudo defaults write com.apple.Safari ShowFullURLInSmartSearchField -bool true
	sudo defaults write com.apple.Safari HomePage -string "about:blank"
	sudo defaults write com.apple.Safari AutoOpenSafeDownloads -bool false
	sudo defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2BackspaceKeyNavigationEnabled -bool true
	sudo defaults write com.apple.Safari ShowFavoritesBar -bool false
	sudo defaults write com.apple.Safari ShowSidebarInTopSites -bool false
	sudo defaults write com.apple.Safari DebugSnapshotsUpdatePolicy -int 2
	sudo defaults write com.apple.Safari IncludeInternalDebugMenu -bool true
	sudo defaults write com.apple.Safari FindOnPageMatchesWordStartsOnly -bool false
	sudo defaults write com.apple.Safari ProxiesInBookmarksBar "()"
	sudo defaults write com.apple.Safari IncludeDevelopMenu -bool true
	sudo defaults write com.apple.Safari WebKitDeveloperExtrasEnabledPreferenceKey -bool true
	sudo defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2DeveloperExtrasEnabled -bool true
	sudo defaults write NSGlobalDomain WebKitDeveloperExtras -bool true
	sudo defaults write com.apple.Safari WebContinuousSpellCheckingEnabled -bool true
	sudo defaults write com.apple.Safari WebAutomaticSpellingCorrectionEnabled -bool false
	sudo defaults write com.apple.Safari AutoFillFromAddressBook -bool false
	sudo defaults write com.apple.Safari AutoFillPasswords -bool false
	sudo defaults write com.apple.Safari AutoFillCreditCardData -bool false
	sudo defaults write com.apple.Safari AutoFillMiscellaneousForms -bool false
	sudo defaults write com.apple.Safari WarnAboutFraudulentWebsites -bool true
	sudo defaults write com.apple.Safari WebKitPluginsEnabled -bool false
	sudo defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2PluginsEnabled -bool false
	sudo defaults write com.apple.Safari WebKitJavaScriptCanOpenWindowsAutomatically -bool false
	sudo defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2JavaScriptCanOpenWindowsAutomatically -bool false
	sudo defaults write com.apple.Safari SendDoNotTrackHTTPHeader -bool true
	sudo defaults write com.apple.Safari InstallExtensionUpdatesAutomatically -bool true

	# -------------------------------------------------------------------------
	# Mail
	# -------------------------------------------------------------------------
	print_step "Configuring Mail..."

	sudo defaults write com.apple.mail DisableReplyAnimations -bool true
	sudo defaults write com.apple.mail DisableSendAnimations -bool true
	sudo defaults write com.apple.mail AddressesIncludeNameOnPasteboard -bool false
	sudo defaults write com.apple.mail NSUserKeyEquivalents -dict-add "Send" "@\U21a9"
	sudo defaults write com.apple.mail DraftsViewerAttributes -dict-add "DisplayInThreadedMode" -string "yes"
	sudo defaults write com.apple.mail DraftsViewerAttributes -dict-add "SortedDescending" -string "yes"
	sudo defaults write com.apple.mail DraftsViewerAttributes -dict-add "SortOrder" -string "received-date"
	sudo defaults write com.apple.mail DisableInlineAttachmentViewing -bool true
	sudo defaults write com.apple.mail SpellCheckingBehavior -string "NoSpellCheckingEnabled"

	# -------------------------------------------------------------------------
	# Terminal & iTerm
	# -------------------------------------------------------------------------
	print_step "Configuring Terminal..."

	defaults write com.apple.terminal StringEncodings -array 4
	defaults write com.apple.terminal SecureKeyboardEntry -bool true
	defaults write com.apple.Terminal ShowLineMarks -int 0
	defaults write com.googlecode.iterm2 PromptOnQuit -bool false

	# -------------------------------------------------------------------------
	# Other Apps
	# -------------------------------------------------------------------------
	print_step "Configuring other applications..."

	# Activity Monitor
	defaults write com.apple.ActivityMonitor OpenMainWindow -bool true
	defaults write com.apple.ActivityMonitor IconType -int 5
	defaults write com.apple.ActivityMonitor ShowCategory -int 0
	defaults write com.apple.ActivityMonitor SortColumn -string "CPUUsage"
	defaults write com.apple.ActivityMonitor SortDirection -int 0

	# TextEdit
	defaults write com.apple.TextEdit RichText -int 0
	defaults write com.apple.TextEdit PlainTextEncoding -int 4
	defaults write com.apple.TextEdit PlainTextEncodingForWrite -int 4

	# Disk Utility
	defaults write com.apple.DiskUtility DUDebugMenuEnabled -bool true
	defaults write com.apple.DiskUtility advanced-image-options -bool true

	# Time Machine
	defaults write com.apple.TimeMachine DoNotOfferNewDisksForBackup -bool true

	# Photos
	defaults -currentHost write com.apple.ImageCapture disableHotPlug -bool true

	# Messages
	defaults write com.apple.messageshelper.MessageController SOInputLineSettings -dict-add "automaticEmojiSubstitutionEnablediMessage" -bool false
	defaults write com.apple.messageshelper.MessageController SOInputLineSettings -dict-add "automaticQuoteSubstitutionEnabled" -bool false
	defaults write com.apple.messageshelper.MessageController SOInputLineSettings -dict-add "continuousSpellCheckingEnabled" -bool false

	# Chrome
	defaults write com.google.Chrome AppleEnableSwipeNavigateWithScrolls -bool false
	defaults write com.google.Chrome AppleEnableMouseSwipeNavigateWithScrolls -bool false
	defaults write com.google.Chrome DisablePrintPreview -bool false
	defaults write com.google.Chrome PMPrintingExpandedStateForPrint2 -bool true

	# Mac App Store
	defaults write com.apple.appstore WebKitDeveloperExtras -bool true
	defaults write com.apple.appstore ShowDebugMenu -bool true
	defaults write com.apple.SoftwareUpdate AutomaticCheckEnabled -bool true
	defaults write com.apple.SoftwareUpdate ScheduleFrequency -int 1
	defaults write com.apple.SoftwareUpdate AutomaticDownload -int 1
	defaults write com.apple.SoftwareUpdate CriticalUpdateInstall -int 1
	defaults write com.apple.SoftwareUpdate ConfigDataInstall -int 1
	defaults write com.apple.commerce AutoUpdate -bool true
	defaults write com.apple.commerce AutoUpdateRestartRequired -bool true
}

# =============================================================================
# Restart Affected Applications
# =============================================================================

restart_apps() {
	print_section "Restarting Affected Applications"

	for app in "Activity Monitor" \
		"cfprefsd" \
		"Dock" \
		"Finder" \
		"Google Chrome" \
		"Mail" \
		"Messages" \
		"Photos" \
		"Safari" \
		"SystemUIServer" \
		"Terminal"; do
		killall "${app}" &> /dev/null || true
	done
}

# =============================================================================
# Main
# =============================================================================

main() {
	echo ""
	echo "╔═══════════════════════════════════════════════════════════════════════════╗"
	echo "║                           macOS Setup Script                              ║"
	echo "╚═══════════════════════════════════════════════════════════════════════════╝"

	if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
		sync_dotfiles
		install_homebrew_packages
		configure_macos
		restart_apps
	else
		read -q "REPLY?This will overwrite files and change system settings. Continue? (y/n) "
		echo ""
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			sync_dotfiles
			install_homebrew_packages
			configure_macos
			restart_apps
		else
			echo "Aborted."
			exit 1
		fi
	fi

	print_section "Setup Complete"
	echo "Some changes require a logout or restart to take effect."
	echo ""

	# Source the new shell config
	source ~/.zshrc
}

main "$@"

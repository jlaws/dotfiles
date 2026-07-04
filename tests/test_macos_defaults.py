"""Tests for macos_setup.macos_defaults pure logic."""

import unittest

from macos_setup.macos_defaults import (
    SETTINGS,
    Setting,
    applicable,
    build_write_args,
    domain_snapshot_name,
    guard_matches,
    merge_revert,
    version_gte,
)


class BuildWriteArgsTests(unittest.TestCase):
    def test_bool_true(self):
        s = Setting("com.apple.dock", "autohide", "bool", True)
        self.assertEqual(build_write_args(s), ["-bool", "true"])

    def test_bool_false(self):
        s = Setting("com.apple.dock", "launchanim", "bool", False)
        self.assertEqual(build_write_args(s), ["-bool", "false"])

    def test_int(self):
        s = Setting("com.apple.dock", "tilesize", "int", 36)
        self.assertEqual(build_write_args(s), ["-int", "36"])

    def test_float(self):
        s = Setting("com.apple.dock", "autohide-delay", "float", 0.0)
        self.assertEqual(build_write_args(s), ["-float", "0.0"])

    def test_string(self):
        s = Setting("com.apple.dock", "mineffect", "string", "scale")
        self.assertEqual(build_write_args(s), ["-string", "scale"])

    def test_raw_passes_value_through(self):
        s = Setting("com.apple.finder", "FXInfoPanesExpanded", "raw", ["-dict", "General", "-bool", "true"])
        self.assertEqual(build_write_args(s), ["-dict", "General", "-bool", "true"])


class GuardMatchesTests(unittest.TestCase):
    def test_equal_matches(self):
        self.assertTrue(guard_matches(36, 36))

    def test_unequal_does_not_match(self):
        self.assertFalse(guard_matches(48, 36))

    def test_missing_current_does_not_match(self):
        self.assertFalse(guard_matches(None, 36))


class MergeRevertTests(unittest.TestCase):
    def test_scalar_restored_to_original(self):
        result = merge_revert(
            live={"tilesize": 36, "other": 1},
            saved={"tilesize": 48, "other": 1},
            keys=["tilesize"],
            applied={"tilesize": 36},
        )
        self.assertEqual(result.restore, {"tilesize": 48})
        self.assertEqual(result.delete, [])
        self.assertEqual(result.skipped, [])

    def test_absent_original_is_deleted(self):
        result = merge_revert(
            live={"AppleShowAllFiles": True},
            saved={},
            keys=["AppleShowAllFiles"],
            applied={"AppleShowAllFiles": True},
        )
        self.assertEqual(result.delete, ["AppleShowAllFiles"])
        self.assertEqual(result.restore, {})

    def test_user_modified_is_skipped(self):
        result = merge_revert(
            live={"tilesize": 64},
            saved={"tilesize": 48},
            keys=["tilesize"],
            applied={"tilesize": 36},
        )
        self.assertEqual(result.restore, {})
        self.assertEqual(result.delete, [])
        self.assertEqual(result.skipped, ["tilesize"])

    def test_key_deleted_by_user_is_skipped(self):
        result = merge_revert(
            live={},
            saved={"tilesize": 48},
            keys=["tilesize"],
            applied={"tilesize": 36},
        )
        self.assertEqual(result.skipped, ["tilesize"])
        self.assertEqual(result.restore, {})

    def test_native_list_value_restored(self):
        result = merge_revert(
            live={"orderedItems": ["a", "b"]},
            saved={"orderedItems": ["x"]},
            keys=["orderedItems"],
            applied={"orderedItems": ["a", "b"]},
        )
        self.assertEqual(result.restore, {"orderedItems": ["x"]})
        self.assertEqual(result.delete, [])


class VersionTests(unittest.TestCase):
    def test_gte_true(self):
        self.assertTrue(version_gte("14.1", "13.0"))

    def test_gte_equal(self):
        self.assertTrue(version_gte("13.0", "13.0"))

    def test_gte_false(self):
        self.assertFalse(version_gte("12.6", "13.0"))

    def test_applicable_filters_by_min_version(self):
        always = Setting("d", "k1", "bool", True)
        gated = Setting("d", "k2", "bool", True, min_version="14.0")
        result = applicable([always, gated], "13.0")
        self.assertEqual([s.key for s in result], ["k1"])

    def test_applicable_includes_when_version_met(self):
        gated = Setting("d", "k2", "bool", True, min_version="14.0")
        result = applicable([gated], "14.0")
        self.assertEqual([s.key for s in result], ["k2"])


class SnapshotNameTests(unittest.TestCase):
    def test_plain_domain(self):
        self.assertEqual(domain_snapshot_name("com.apple.dock"), "com.apple.dock.plist")

    def test_global_domain(self):
        self.assertEqual(domain_snapshot_name("NSGlobalDomain"), "NSGlobalDomain.plist")

    def test_path_domain_is_flattened(self):
        self.assertEqual(
            domain_snapshot_name("/Library/Preferences/com.apple.loginwindow"),
            "Library_Preferences_com.apple.loginwindow.plist",
        )


class RegistryTests(unittest.TestCase):
    def test_every_setting_builds_valid_args(self):
        for s in SETTINGS:
            with self.subTest(domain=s.domain, key=s.key):
                args = build_write_args(s)
                self.assertIsInstance(args, list)
                self.assertTrue(all(isinstance(a, str) for a in args))

    def test_every_scope_is_known(self):
        for s in SETTINGS:
            with self.subTest(domain=s.domain, key=s.key):
                self.assertIn(s.scope, {"user", "sudo", "host"})


if __name__ == "__main__":
    unittest.main()

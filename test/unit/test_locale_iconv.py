"""Platform availability and exact signatures of locale/iconv bindings."""

import unittest

from pythoc import i8, i32, ptr, void
from test.utils.libc_platform import platform_bindings, signature_types


class TestLocaleIconv(unittest.TestCase):
    def test_locale_api_platforms(self):
        for platform in ('darwin', 'linux', 'win32'):
            with self.subTest(platform=platform):
                with platform_bindings(platform, ('locale',)) as modules:
                    locale = modules['locale']
                    self.assertTrue(locale.localeconv._is_extern)
                    self.assertIs(locale.localeconv.return_type.pointee_type,
                                  void)
                    self.assertIn('localeconv', locale.__all__)
                    self.assertEqual(hasattr(locale, 'nl_langinfo'),
                                     platform != 'win32')
                    self.assertEqual('CODESET' in locale.__all__,
                                     platform != 'win32')
                    self.assertEqual('LC_MESSAGES' in locale.__all__,
                                     platform != 'win32')
                    for name in locale.__all__:
                        self.assertTrue(hasattr(locale, name), name)

    def test_iconv_signatures_and_libraries(self):
        for platform, library in (('darwin', 'iconv'), ('linux', 'c')):
            with self.subTest(platform=platform):
                with platform_bindings(platform, ('stddef', 'iconv')) as modules:
                    module = modules['iconv']
                    size_t = modules['stddef'].size_t
                    self.assertIs(module.iconv_t.pointee_type, void)
                    self.assertIs(module.iconv.return_type, size_t)
                    parameters = [
                        ('cd', module.iconv_t), ('inbuf', ptr[ptr[i8]]),
                        ('inbytesleft', ptr[size_t]), ('outbuf', ptr[ptr[i8]]),
                        ('outbytesleft', ptr[size_t]),
                    ]
                    self.assertEqual(signature_types(module.iconv.param_types),
                                     signature_types(parameters))
                    self.assertIs(module.iconv_close.return_type, i32)
                    for name in ('iconv_open', 'iconv', 'iconv_close'):
                        function = getattr(module, name)
                        self.assertTrue(function._is_extern)
                        self.assertEqual(function.lib, library)

    def test_windows_iconv_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, '<iconv.h>'):
            with platform_bindings('win32', ('iconv',)):
                self.fail('Windows cannot use iconv from the C runtime')


if __name__ == "__main__":
    unittest.main()

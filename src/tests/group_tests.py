from group_data import *
import unittest

class GroupTests(unittest.TestCase):
    def test_cyclic(self):
        self.assertEqual(len(Z(5)), 5)

    def test_symmetric(self):
        self.assertEqual(len(S(5)), 120)

    def test_properties(self):
        g = S(4)
        self.assertTrue(g.is_group())
        self.assertTrue(g.is_latin())
        self.assertFalse(g.is_abelian())

        g = clamped(10)
        self.assertFalse(g.is_group())

        g = Z(3, 4)
        self.assertTrue(g.is_group())
        self.assertTrue(g.is_latin())
        self.assertTrue(g.is_abelian())

    def test_fp_parse(self):
        g = Z(100)
        cases = [
            # Remember GAP is 1-indexed
            ('f1', 0),
            ('f2*f3', 1 + 2),
            ('f2^4*f3', 1 * 4 + 2),
            ('f2^4*f3*f7*f3^-1*f8^2', 1 * 4 + 2 + 6 - 2 + 7 * 2),
            ('(f2*f4*f5^3)^3*(f3*f5)^-2', (1 + 3 + 4 * 3) * 3 + (2 + 4) * -2),
            ('(f3*(f4*f7)^2*f3)^-1*f5', (2 + (3 + 6) * 2 + 2) * -1 + 4),
        ]
        for fp_elem, idx in cases:
            self.assertEqual(g.fp_elem_to_elem(fp_elem), idx % 100)

    def test_char_table_V4(self):
        char_table = Z(2, 2).get_char_table().real.round().int().tolist()
        correct = \
            [
                [1, 1, 1, 1],
                [1, 1, -1, -1],
                [1, -1, 1, -1],
                [1, -1, -1, 1],
            ]
        # Ordering is arbitrary, so just check that they're same as sets
        self.assertEqual(
            set(map(frozenset, char_table)),
            set(map(frozenset, correct))
        )

    def test_char_table_S5(self):
        char_table = S(5).get_char_table().real.round().int().tolist()
        correct = \
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, 1, -1, -1, 1],
                [4, 2, 0, 1, -1, 0, -1],
                [4, -2, 0, 1, 1, 0, -1],
                [5, 1, 1, -1, 1, -1, 0],
                [5, -1, 1, -1, -1, 1, 0],
                [6, 0, -2, 0, 0, 0, 1],
            ]
        # Ordering is arbitrary, so just check that they're same as sets
        self.assertEqual(
            set(map(frozenset, char_table)),
            set(map(frozenset, correct))
        )

    def test_subgroups(self):
        g = S(4)
        subgroups = g.get_subgroups()
        self.assertEqual(len(subgroups), 30)
        self.assertEqual(
            {len(h) for h in subgroups}, 
            # See https://groupprops.subwiki.org/wiki/Subgroup_structure_of_symmetric_group:S4
            set([1] + [2] * 9 + [4] * 7 + [8] * 3 + [3] * 4 + [6] * 4  + [12] + [24])
        )

        # This takes ~1 min
        # g = S(5)
        # subgroups = g.get_subgroups()
        # self.assertEqual(len(subgroups), 156)

    def test_complex_irreps(self):
        g = Z(4)
        irreps = g.get_irreps()
        real_irreps = g.get_irreps(real=True)
        self.assertEqual(
            set(irreps.keys()),
            {'1d-0', '1d-1', '1d-2', '1d-3'}
        )
        self.assertEqual(
            set(real_irreps.keys()),
            {'1d-0', '1d-1', '2d-0', '2d-1'}
        )
        for irrep in irreps.values():
            self.assertTrue(g.is_irrep(irrep))
        for irrep in real_irreps.values():
            self.assertTrue(g.is_irrep(irrep))

        g = D(8)
        irreps = g.get_irreps()
        real_irreps = g.get_irreps(real=True)
        self.assertEqual(
            set(irreps.keys()),
            {'1d-0', '1d-1', '1d-2', '1d-3', '2d-0', '2d-1', '2d-2'}
        )
        self.assertEqual(
            set(real_irreps.keys()),
            {'1d-0', '1d-1', '1d-2', '1d-3', '2d-0', '2d-1', '2d-2'}
        )
        for irrep in irreps.values():
            self.assertTrue(g.is_irrep(irrep))
        for irrep in real_irreps.values():
            self.assertTrue(g.is_irrep(irrep))

        g = gapS(5)
        irreps = g.get_irreps()
        real_irreps = g.get_irreps(True)
        self.assertEqual(
            set(irreps.keys()),
            {'1d-0', '1d-1', '4d-0', '4d-1', '5d-0', '5d-1', '6d-0'}
        )
        self.assertEqual(
            set(real_irreps.keys()),
            {'1d-0', '1d-1', '4d-0', '4d-1', '5d-0', '5d-1', '6d-0'}
        )
        for irrep in irreps.values():
            self.assertTrue(g.is_irrep(irrep))
        for irrep in real_irreps.values():
            self.assertTrue(g.is_irrep(irrep))

        g = smallgrp(110, 1)
        irreps = g.get_irreps()
        real_irreps = g.get_irreps(True)
        self.assertEqual(
            set(irreps.keys()),
            {'1d-0', '1d-1', '1d-2', '1d-3', '1d-4', '1d-5', '1d-6', '1d-7', '1d-8', '1d-9', '10d-0'}
        )
        self.assertEqual(
            set(real_irreps.keys()),
            {'1d-0', '1d-1', '2d-0', '2d-1', '2d-2', '2d-3', '2d-4', '2d-5', '2d-6', '2d-7', '10d-0'}
        )
        for irrep in irreps.values():
            self.assertTrue(g.is_irrep(irrep))
        for irrep in real_irreps.values():
            self.assertTrue(g.is_irrep(irrep))

    def test_frobenius_schur(self):
        g = smallgrp(110, 1)
        irreps = g.get_irreps(real=False)
        indicators = {g.get_frobenius_schur(irrep) for irrep in irreps.values()}
        self.assertEqual(
            indicators,
            set([1] * 3 + [0] * 8)
        )


if __name__ == '__main__':
    unittest.main()
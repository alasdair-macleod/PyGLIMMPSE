from unittest import TestCase
from unittest.mock import patch

from pyglimmpse.constants import Constants
from pyglimmpse.probf import probf, _normal_approximation, _get_zscore, _tiku_approximation, _nonadjusted


class TestProbf(TestCase):

    @patch('pyglimmpse.probf._nonadjusted')
    def test_probf_nonadjusted(self, mock):
        """probf should call the nonadjusted method for these input values"""
        mock.return_value = (None, None)
        probf(0, 0, 0, 0)
        assert mock.called

    @patch('pyglimmpse.probf._nonadjusted')
    def test_probf_nonadjusted_path_two(self, mock):
        """probf should call the nonadjusted method for these input values"""
        mock.return_value = (None, None)
        assert(not mock.called)
        probf(0, 10**5, 9, 10**5.9)
        assert mock.called

    @patch('pyglimmpse.probf._tiku_approximation')
    def test_probf_tiku_approx(self, mock):
        """probf should call the _tiku_approximation method for these input values"""
        mock.return_value = (None, None)
        assert (not mock.called)
        probf(0, 10**7, 10, 10)
        assert mock.called

    @patch('pyglimmpse.probf._normal_approximation')
    @patch('pyglimmpse.probf._get_zscore')
    def test_probf_norm_approx(self, zmock, mock):
        """probf should call the normal approximation method for these input values"""
        mock.return_value = (None, None)
        probf(0, 10 ** 9.5, 0, 0)
        assert mock.called

    @patch('scipy.stats.norm.cdf')
    def test__normal_approximation_fmethod_three(self, mock):
        """Should calculate prob using norm.cdf for |zcsore| < 6 and give fmethod 3"""
        expected = (0, Constants.FMETHOD_NORMAL_SM)
        mock.return_value = 0
        actual = _normal_approximation(0)
        assert mock.called
        self.assertEqual(expected, actual)

    @patch('scipy.stats.norm.cdf')
    def test__normal_approximation_fmethod_four(self, mock):
        """Should not call norm.cdf for |zcsore| > 6
        and return prob 0 for -ve values or 1 for +ve values"""
        expected = (0, Constants.FMETHOD_NORMAL_LR)
        mock.return_value = 0
        actual = _normal_approximation(-7)
        assert not mock.called
        assert actual == expected

        expected = (1, Constants.FMETHOD_NORMAL_LR)
        mock.return_value = 0
        actual = _normal_approximation(7)
        assert not mock.called
        assert actual == expected

    # TODO raise our own error here and handle sensibly
    @patch('scipy.stats.norm.cdf')
    def test_normal_approximation_method_four(self, mock):
        """Should raise an informative error when zscore = 6"""
        expected = (-1, None)
        with self.assertRaises(UnboundLocalError):  # Let's define our own exception types here'
            actual = _normal_approximation(6)
            self.assertEquals(actual, expected)

    def test_get_zscore(self):
        """Should return the expected value for normal data"""
        expected = -0.39007506867134967
        actual = _get_zscore(1, 1, 1, 1)
        self.assertEqual(expected, actual)

    # TODO raise our own error here and handle sensibly
    def test_get_zscore_div_zero(self):
        """Should raise informative exception for values that would result in div/0 error"""
        with self.assertRaises(ZeroDivisionError):
            actual = _get_zscore(0, 1, 1, 0)
        with self.assertRaises(ZeroDivisionError):
            actual = _get_zscore(1, 0, 1, 1)

    @patch('scipy.special.ncfdtr')
    def test__tiku_approximation(self, mock):
        """Should return fmethod 2 and call special.ncfdtr for normal input"""
        expected = (2, 0)
        mock.return_value = 0
        assert not mock.called
        actual = _tiku_approximation(1, -2, 2, 0)
        self.assertEquals(expected, actual)

    # TODO raise our own error here and handle sensibly
    @patch('scipy.special.ncfdtr')
    def test__tiku_approximation(self, mock):
        """Should return fmethod 2 and call special.ncfdtr for normal input"""
        mock.return_value = 0
        assert not mock.called
        with self.assertRaises(ZeroDivisionError):
            actual = _tiku_approximation(0, 0, 0, 0)

    @patch('scipy.special.ncfdtr')
    def test__nonadjusted(self, mock):
        """Should calculate prob using special.ncfdtr and return fmethod 1"""
        expected = (0, Constants.FMETHOD_NOAPPROXIMATION)
        mock.return_value = 0
        actual = _nonadjusted(0, 0, 0, 0)
        self.assertEqual(expected, actual)

    def test__probf(self):
        """Should have the same prob and fmethod as in IML"""
        expected = (0.7853726, Constants.FMETHOD_NOAPPROXIMATION)
        result = probf(1.96, 0.5, 3, 0)
        actual = (round(result[0], 7), result[1])
        self.assertEqual(expected, actual)

    def test__probf_2(self):
        """Should have the same prob and fmethod as in IML"""
        one_01 = probf(1, 1, 10, 0)
        one_02 = probf(1, 1, 10, 1)
        one_03 = probf(1, 1, 10, 5)
        one_04 = probf(1, 1, 10, 10)
        one_05 = probf(1, 1, 10, 100)
        two_01 = probf(1, 1, 50, 0)
        two_02 = probf(1, 1, 50, 1)
        two_03 = probf(1, 1, 50, 5)
        two_04 = probf(1, 1, 50, 10)
        two_05 = probf(1, 1, 50, 100)
        three_01 = probf(1, 1, 100, 0)
        three_02 = probf(1, 1, 100, 1)
        three_03 = probf(1, 1, 100, 5)
        three_04 = probf(1, 1, 100, 10)
        three_05 = probf(1, 1, 100, 100)

        two_one_01 = probf(1, 1, 10, 0)
        two_one_02 = probf(1, 1, 10, 1)
        two_one_03 = probf(1, 1, 10, 5)
        two_one_04 = probf(1, 1, 10, 10)
        two_one_05 = probf(1, 1, 10, 100)
        two_two_01 = probf(1, 1, 50, 0)
        two_two_02 = probf(1, 1, 50, 1)
        two_two_03 = probf(1, 1, 50, 5)
        two_two_04 = probf(1, 1, 50, 10)
        two_two_05 = probf(1, 1, 50, 100)
        two_three_01 = probf(1, 1, 100, 0)
        two_three_02 = probf(1, 1, 100, 1)
        two_three_03 = probf(1, 1, 100, 5)
        two_three_04 = probf(1, 1, 100, 10)
        two_three_05 = probf(1, 1, 100, 100)

        three_one_01 = probf(1, 1, 10, 0)
        three_one_02 = probf(1, 1, 10, 1)
        three_one_03 = probf(1, 1, 10, 5)
        three_one_04 = probf(1, 1, 10, 10)
        three_one_05 = probf(1, 1, 10, 100)
        three_two_01 = probf(1, 1, 50, 0)
        three_two_02 = probf(1, 1, 50, 1)
        three_two_03 = probf(1, 1, 50, 5)
        three_two_04 = probf(1, 1, 50, 10)
        three_two_05 = probf(1, 1, 50, 100)
        three_three_01 = probf(1, 1, 100, 0)
        three_three_02 = probf(1, 1, 100, 1)
        three_three_03 = probf(1, 1, 100, 5)
        three_three_04 = probf(1, 1, 100, 10)
        three_three_05 = probf(1, 1, 100, 100)

        x = 1

    def test_probf_chi2(self):
        """ Should have the correct power and fmethod """
        expected = (0.031833, Constants.FMETHOD_CHI2)
        result = probf(66.3490, 1, 10**10, 100)
        actual = (round(result[0], 6), result[1])
        self.assertEqual(expected, actual)




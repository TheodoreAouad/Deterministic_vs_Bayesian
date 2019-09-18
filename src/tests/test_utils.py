from src.utils import split_multiple_sep


class TestSplitMultipleSep:

    @staticmethod
    def test_one_sep():
        s1 = 'varation ratio'
        s2 = 'varation-ratio'

        split1 = split_multiple_sep(s1, [' '])
        split2 = split_multiple_sep(s2, ['-'])
        split3 = split_multiple_sep(s2, [' '])

        assert split1 == ['varation', 'ratio']
        assert split2 == ['varation', 'ratio']
        assert split3 == ['varation-ratio']

    @staticmethod
    def test_multiple_sep():
        s1 = 'varation ratio'
        s2 = 'varation-ratio'

        split1 = split_multiple_sep(s1, [' ', '-'])
        split2 = split_multiple_sep(s2, [' ', '-'])

        assert split1 == ['varation', 'ratio']
        assert split2 == ['varation', 'ratio']

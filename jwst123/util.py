"""Small shared helpers (CLI banners, coordinate parsing)."""

from astropy import units as u
from astropy.coordinates import SkyCoord


def make_banner(message):
    print('\n\n' + message + '\n' + '#' * 80 + '\n' + '#' * 80 + '\n\n')


def is_number(num):
    try:
        float(num)
    except (TypeError, ValueError):
        return False
    return True


def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
            (':' not in str(ra) and ':' not in str(dec))):
        print(f'ERROR: cannot interpret: {ra} {dec}')
        return None

    if ':' in str(ra) and ':' in str(dec):
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        return SkyCoord(ra, dec, frame='icrs', unit=unit)
    except ValueError:
        print(f'ERROR: Cannot parse coordinates: {ra} {dec}')
        return None

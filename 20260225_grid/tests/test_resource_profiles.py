from __future__ import annotations

import unittest

from scripts.resource_profiles import (
    RealResourceSettings,
    _generic_wind_power_fraction,
    parse_nsrdb_csv,
    parse_wtk_csv,
    solar_generation_profile_from_nsrdb,
    wind_generation_profile_from_wtk,
)


NSRDB_SAMPLE = """Source,Location ID,City,State,Country,Latitude,Longitude,Time Zone,Elevation,Local Time Zone,Clearsky DHI Units,Clearsky DNI Units,Clearsky GHI Units,Dew Point Units,DHI Units,DNI Units,GHI Units,Solar Zenith Angle Units,Temperature Units,Pressure Units,Relative Humidity Units,Precipitable Water Units,Wind Direction Units,Wind Speed Units,Version
NSRDB,949192,-,-,-,40.14,-105.23,-7,1645,-7,w/m2,w/m2,w/m2,c,w/m2,w/m2,w/m2,Degree,c,mbar,%,cm,Degrees,m/s,4.0.0
Year,Month,Day,Hour,Minute,GHI,DNI,DHI,Temperature,Wind Speed
2020,1,1,0,30,0,0,0,-2,6.1
2020,1,1,8,30,149,697,32,1.2,6.1
2020,1,1,13,30,430,945,48,5.0,4.9
"""

WTK_SAMPLE = """SiteID,853222,Site Timezone,-7,Data Timezone,-7,Longitude,-105.249176025,Latitude,40.1296768188
Year,Month,Day,Hour,Minute,wind speed at 100m (m/s),wind direction at 100m (deg),air temperature at 100m (C),air pressure at 100m (Pa)
2014,1,1,0,30,1.43,311.6,4.65,81940
2014,1,1,12,30,5.02,78.68,1.71,82470
2014,1,1,13,30,12.69,260.52,8.16,82380
2014,1,1,14,30,30.20,275.86,9.86,81050
"""


class ResourceProfilesTest(unittest.TestCase):
    def test_parse_nsrdb_csv_shape(self) -> None:
        metadata, rows = parse_nsrdb_csv(NSRDB_SAMPLE)
        self.assertEqual(metadata["Source"], "NSRDB")
        self.assertEqual(metadata["Location ID"], "949192")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["ghi"], 0.0)
        self.assertEqual(rows[1]["dni"], 697.0)
        self.assertEqual(rows[2]["temperature_c"], 5.0)

    def test_parse_wtk_csv_shape(self) -> None:
        metadata, rows = parse_wtk_csv(WTK_SAMPLE)
        self.assertEqual(metadata["SiteID"], "853222")
        self.assertEqual(metadata["Site Timezone"], "-7")
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]["wind_speed_ms"], 1.43)
        self.assertAlmostEqual(rows[2]["wind_speed_ms"], 12.69)
        self.assertAlmostEqual(rows[3]["pressure_pa"], 81050.0)

    def test_solar_profile_uses_ghi_and_zeroes_night(self) -> None:
        _, rows = parse_nsrdb_csv(NSRDB_SAMPLE)
        profile = solar_generation_profile_from_nsrdb(rows, RealResourceSettings())
        self.assertEqual(len(profile), 3)
        self.assertEqual(profile[0], 0.0)
        self.assertGreater(profile[2], profile[1])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in profile))

    def test_generic_wind_curve_shape(self) -> None:
        settings = RealResourceSettings()
        self.assertEqual(_generic_wind_power_fraction(2.0, settings), 0.0)
        self.assertGreater(_generic_wind_power_fraction(6.0, settings), 0.0)
        self.assertEqual(_generic_wind_power_fraction(12.0, settings), 1.0)
        self.assertEqual(_generic_wind_power_fraction(30.0, settings), 0.0)

    def test_wind_profile_applies_losses_and_cutout(self) -> None:
        _, rows = parse_wtk_csv(WTK_SAMPLE)
        profile = wind_generation_profile_from_wtk(rows, RealResourceSettings())
        self.assertEqual(len(profile), 4)
        self.assertEqual(profile[0], 0.0)
        self.assertGreater(profile[1], 0.0)
        self.assertAlmostEqual(profile[2], 0.85)
        self.assertEqual(profile[3], 0.0)


if __name__ == "__main__":
    unittest.main()

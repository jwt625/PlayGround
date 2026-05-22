# IEEE P802.3dj Metadata Fix Checklist

Manual review notes for cached presentation metadata in `ieee802_3dj_browser/metadata/talks.json`.

Reviewed fields: title, presenter, affiliation. The source checks were the cached meeting index rows plus the first-page extracted text in `ieee802_3dj_browser/extracted_text/`.

## Missing Presenter/Affiliation

- [x] `23_07__mellitz_3dj_01a_2307` `mellitz_3dj_01a_2307.pdf` - missing presenters/affiliations. First slide: Richard Mellitz - Samtec; Kent Lusted - Intel; Matt Brown - Alphawave Semi.
- [x] `23_09__lusted_3dj_05_2309` `lusted_3dj_05_2309.pdf` - missing presenters/affiliations. First slide: Kent Lusted - Intel Corporation; Matt Brown - Alphawave Semi.
- [x] `24_07__Vince_Ferretti_and_Kazuhide_Nakajima` `Vince Ferretti and Kazuhide Nakajima.pdf` - missing presenters/affiliations. First slide: Vince Ferretti - Corning / Associate Rapporteur ITU-T SG15 Q5; Kazuhide Nakajima - NTT / Rapporteur ITU-T SG15 Q5.
- [x] `24_09__mi_3dj_02a_2409` `mi_3dj_02a_2409.pdf` - missing presenter/affiliation. First slide: Guangcan Mi - Huawei Technologies Co., Ltd.
- [x] `24_11__maniloff_3dj_01a_2411` `maniloff_3dj_01a_2411.pdf` - missing presenters/affiliations. First slide: Eric Maniloff and Riyaz Jamal - Ciena; Xiang Liu and Qirui Fan - Huawei.

## Non-Presentation Or Redacted Files With Missing People

- [x] `23_07__IEEE_802d3_to_OIF_3dj_2307_draft_Redacted` `IEEE_802d3_to_OIF_3dj_2307_draft_Redacted.pdf` - missing people; extracted text has no usable author page.
- [x] `25_09__IEEE_802d3_to_ITU_1a_3dj_2509_draft_Redacted` `IEEE_802d3_to_ITU_1a_3dj_2509_draft_Redacted.pdf` - missing people; extracted text has no usable author page.
- [x] `25_09__IEEE_802d3_to_ITU_3_3dj_2509_draft_Redacted` `IEEE_802d3_to_ITU_3_3dj_2509_draft_Redacted.pdf` - missing people; liaison source text lists IEEE 802.3 Working Group rather than individual presenters.

## HTML Markup Leaked Into People Fields

- [x] `26_03__alloin_3dj_01a_2603` `alloin_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__alloin_3dj_02a_2603` `alloin_3dj_02a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__bruckman_3dj_01_2603` `bruckman_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter.
- [x] `26_03__bruckman_3dj_02a_2603` `bruckman_3dj_02a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter.
- [x] `26_03__cole_3dj_01_2603` `cole_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter and first affiliation.
- [x] `26_03__dekoos_3dj_01_2603` `dekoos_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__dudek_3dj_01_2603` `dudek_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter.
- [x] `26_03__fan_3dj_01a_2603` `fan_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__ghiasi_3dj_01a_2603` `ghiasi_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__ghiasi_3dj_02a_2603` `ghiasi_3dj_02a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__he_3dj_01_2603` `he_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__lusted_3dj_01_2603` `lusted_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__lusted_3dj_02a_2603` `lusted_3dj_02a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__maniloff_3dj_01a_2603` `maniloff_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__mascitto_3dj_01a_2603` `mascitto_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.
- [x] `26_03__mellitz_3dj_01b_2603` `mellitz_3dj_01b_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__mellitz_3dj_02_2603` `mellitz_3dj_02_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__mi_3dj_01a_2603` `mi_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.
- [x] `26_03__ran_3dj_01a_2603` `ran_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter.
- [x] `26_03__ran_3dj_03a_2603` `ran_3dj_03a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__ran_3dj_04a_2603` `ran_3dj_04a_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.
- [x] `26_03__slavick_3dj_01_2603` `slavick_3dj_01_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.
- [x] `26_03__slavick_3dj_02_2603` `slavick_3dj_02_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.
- [x] `26_03__slavick_3dj_03_2603` `slavick_3dj_03_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter.
- [x] `26_03__swenson_3dj_01a_2603` `swenson_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__swenson_3dj_02a_2603` `swenson_3dj_02a_2603.pdf` - remove leaked `<!--td ...-->` prefix from presenter and affiliation.
- [x] `26_03__wang_3dj_01a_2603` `wang_3dj_01a_2603.pdf` - remove leaked `<!--td ...-->` prefix from first presenter.

## Wrong Presenter/Affiliation Parsing

- [x] `24_06__brown_3dj_02_2406` `brown_3dj_02_2406.pdf` - presenters parsed as file codes `brown_02`, `brown_02a`; should be Matt Brown - Alphawave Semi and Leon Bruckman - Huawei from first slide.
- [x] `24_06__brown_3dj_02a_2406` `brown_3dj_02a_2406.pdf` - presenters parsed as file codes `brown_02`, `brown_02a`; should be Matt Brown - Alphawave Semi and Leon Bruckman - Huawei from first slide.
- [x] `24_06__brown_3dj_02b_2406` `brown_3dj_02b_2406.pdf` - presenters parsed as file codes `brown_02`, `brown_02a`; should be Matt Brown - Alphawave Semi and Leon Bruckman - Huawei from first slide.
- [x] `24_06__loewenthal_3dj_01a_2406` `loewenthal_3dj_01a_2406.pdf` - presenters parsed as file codes `loewenthal_01`, `loewenthal_02`; affiliation field contains people. Re-parse from first slide.
- [x] `25_07__he_m_3dj_01c_2507` `he_m_3dj_01c_2507.pdf` - presenter parsed as file code `he_m_01`; affiliation field contains Michael He. Re-parse from first slide.
- [x] `25_07__he_x_3dj_01_2507` `he_x_3dj_01_2507.pdf` - presenters parsed as file codes `he_x_01`, `he_x_02`; affiliation field contains Xiang He and Xuebo Wang. Re-parse from first slide.
- [x] `23_05__gui_3dj_01a_2305` `gui_3dj_01a_2305.pdf` - presenter `QianXiang` should be `Qian Xiang`; all three listed presenters are Huawei.
- [x] `23_05__li_3dj_09a_2305` `li_3dj_09a_2305.pdf` - presenter `Ariel Cohen,` has a trailing comma; first slide also includes Megha Shanbhag and Nathan Tracy - TE.
- [x] `23_05__li_3dj_10a_2305` `li_3dj_10a_2305.pdf` - presenter `Ariel Cohen,` has a trailing comma; first slide also includes Megha Shanbhag and Nathan Tracy - TE.
- [x] `25_07__ran_3dj_01_2507` `ran_3dj_01_2507.pdf` - presenter `Sam Kocsi` should be `Sam Kocsis`.
- [x] `25_07__ran_3dj_01a_2507` `ran_3dj_01a_2507.pdf` - presenter `Sam Kocsi` should be `Sam Kocsis`.
- [x] `25_07__ran_3dj_01b_2507` `ran_3dj_01b_2507.pdf` - presenter `Sam Kocsi` should be `Sam Kocsis`.
- [x] `25_07__ran_3dj_01c_2507` `ran_3dj_01c_2507.pdf` - presenter `Sam Kocsi` should be `Sam Kocsis`.
- [x] `26_01__alloin_3dj_01b_2601` `alloin_3dj_01b_2601.pdf` - presenter typos: `Laureent Alloin` -> `Laurent Alloin`; `Eric Manioff` -> `Eric Maniloff`.
- [x] `24_01__liu_3dj_01_2401` `liu_3dj_01_2401.pdf` - presenter `Robert Rodes` should be checked against first slide; likely `Roberto Rodes`.
- [x] `26_01__cole_3dj_01c_2601` `cole_3dj_01c_2601.pdf` - presenter `Robert Rodes` should be checked against first slide; likely `Roberto Rodes`.
- [x] `26_03__cole_3dj_01_2603` `cole_3dj_01_2603.pdf` - presenter `Robert Rodes` should be checked against first slide; likely `Roberto Rodes`.
- [x] `26_05__cole_3dj_01b_2605` `cole_3dj_01b_2605.pdf` - presenter `Robert Rodes` should be checked against first slide; likely `Roberto Rodes`.

## Misspelled Presenter Repeated Across Comment-Resolution Decks

- [x] `24_03__brown_3dj_01_2403` `brown_3dj_01_2403.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_03__brown_3dj_02_2403` `brown_3dj_02_2403.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_05__brown_3dj_01_2405` `brown_3dj_01_2405.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_07__brown_3dj_01_2407` `brown_3dj_01_2407.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_09__brown_3dj_01_2409` `brown_3dj_01_2409.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_09__brown_3dj_02_2409` `brown_3dj_02_2409.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_09__brown_3dj_02a_2409` `brown_3dj_02a_2409.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_09__brown_3dj_02b_2409` `brown_3dj_02b_2409.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_09__brown_3dj_02c_2409` `brown_3dj_02c_2409.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_11__brown_3dj_01a_2411` `brown_3dj_01a_2411.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `24_11__brown_3dj_02e_2411` `brown_3dj_02e_2411.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_01__brown_3dj_01_2501` `brown_3dj_01_2501.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_01__brown_3dj_02c_2501` `brown_3dj_02c_2501.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_01__brown_3dj_02d_2501` `brown_3dj_02d_2501.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_03__brown_3dj_01a_2503` `brown_3dj_01a_2503.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_03__brown_3dj_02b_2503` `brown_3dj_02b_2503.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_09__brown_3dj_02c_2509` `brown_3dj_02c_2509.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_11__brown_3dj_02a_2511` `brown_3dj_02a_2511.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `25_11__brown_3dj_02b_2511` `brown_3dj_02b_2511.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `26_01__brown_3dj_02_2601` `brown_3dj_02_2601.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `26_01__brown_3dj_02a_2601` `brown_3dj_02a_2601.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.
- [x] `26_01__brown_3dj_02b_2601` `brown_3dj_02b_2601.pdf` - `Tom Issenhtuh` should be `Tom Issenhuth`.

## Generic Or Wrong Titles

- [x] `23_07__he_3dj_03b_2307` `he_3dj_03b_2307.pdf` - title is `Update - Inner Codeword Self-sync Proposal`; first slide suggests `Inner Codeword Self-sync Proposal`.
- [x] `24_06__brown_3dj_01a_2406` `brown_3dj_01a_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01b_2406` `brown_3dj_01b_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01c_2406` `brown_3dj_01c_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01d_2406` `brown_3dj_01d_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01e_2406` `brown_3dj_01e_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01f_2406` `brown_3dj_01f_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01g_2406` `brown_3dj_01g_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01h_2406` `brown_3dj_01h_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_01j_2406` `brown_3dj_01j_2406.pdf` - title is `Update`; first slide suggests `P802.3dj D1.0 Comment Resolution Agenda`.
- [x] `24_06__brown_3dj_02a_2406` `brown_3dj_02a_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Common Topics`.
- [x] `24_06__brown_3dj_02b_2406` `brown_3dj_02b_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Common Topics`.
- [x] `24_06__nicholl_3dj_01a_2406` `nicholl_3dj_01a_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Logic Track`.
- [x] `24_06__nicholl_3dj_01b_2406` `nicholl_3dj_01b_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Logic Track`.
- [x] `24_06__ran_3dj_01a_2406` `ran_3dj_01a_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_06__ran_3dj_01b_2406` `ran_3dj_01b_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_06__ran_3dj_01c_2406` `ran_3dj_01c_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_06__ran_3dj_01d_2406` `ran_3dj_01d_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_06__ran_3dj_01e_2406` `ran_3dj_01e_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_06__ran_3dj_01f_2406` `ran_3dj_01f_2406.pdf` - title is `Update`; first slide suggests `802.3dj D1.0 Comment Resolution Electrical Track`.
- [x] `24_09__brown_3dj_02a_2409` `brown_3dj_02a_2409.pdf` - title is `Update (as of 12-Sep-2024)`; first slide suggests `P802.3dj D1.1 Comment Resolution Agenda`.
- [x] `24_09__brown_3dj_02b_2409` `brown_3dj_02b_2409.pdf` - title is `Update (as of 18-Sep-2024)`; first slide suggests `P802.3dj D1.1 Comment Resolution Agenda`.
- [x] `24_09__brown_3dj_02c_2409` `brown_3dj_02c_2409.pdf` - title is `Update (as of 19-Sep-2024)`; first slide suggests `P802.3dj D1.1 Comment Resolution Agenda`.
- [x] `24_09__nicholl_3dj_01a_2409` `nicholl_3dj_01a_2409.pdf` - title is `Update (As of 16 Sept)`; first slide suggests `802.3dj D1.1 Comment Resolution Logic Topics`.
- [x] `24_11__issenhuth_3dj_01a_2411` `issenhuth_3dj_01a_2411.pdf` - title is `Update`; first slide suggests `802.3dj D1.2 Comment Resolution Optical Track`.
- [x] `24_11__ran_3dj_01a_2411` `ran_3dj_01a_2411.pdf` - title is `Update`; first slide suggests `802.3dj D1.2 Comment Resolution Electrical Track`.
- [x] `25_01__brown_3dj_03a_2501` `brown_3dj_03a_2501.pdf` - title is `Update`; first slide suggests `802.3dj D1.3 Comment Resolution Common Track`.
- [x] `25_01__brown_3dj_03b_2501` `brown_3dj_03b_2501.pdf` - title is `Update`; first slide suggests `802.3dj D1.3 Comment Resolution Common Track`.
- [x] `25_01__issenhuth_3dj_01a_2501` `issenhuth_3dj_01a_2501.pdf` - title is `Update`; first slide suggests `802.3dj D1.3 Comment Resolution Optical Track`.
- [x] `25_03__brown_3dj_03a_2503` `brown_3dj_03a_2503.pdf` - title is `Update`; first slide suggests `802.3dj D1.4 Comment Resolution Common Track`.
- [x] `25_03__nicholl_3dj_01a_2503` `nicholl_3dj_01a_2503.pdf` - title is `Update`; first slide suggests `802.3dj D1.4 Comment Resolution Logic Track`.
- [x] `25_07__brown_3dj_03a_2507` `brown_3dj_03a_2507.pdf` - title is `Update - 10 Jul`; first slide suggests `802.3dj D2.0 Comment Resolution Common Track`.
- [x] `25_07__brown_3dj_03b_2507` `brown_3dj_03b_2507.pdf` - title is `Update - 21 Jul`; first slide suggests `802.3dj D2.0 Comment Resolution Common Track`.
- [x] `25_07__brown_3dj_03c_2507` `brown_3dj_03c_2507.pdf` - title is `Update - 28 Jul`; first slide suggests `802.3dj D2.0 Comment Resolution Common Track`.
- [x] `25_07__cole_3dj_01e_2507` `cole_3dj_01e_2507.pdf` - title is `Update`; first slide suggests `Transmitter Functional Symbol Error Mask Test Proposal`.
- [x] `25_07__ran_3dj_01a_2507` `ran_3dj_01a_2507.pdf` - title is `Update`; first slide suggests `802.3dj D2.0 Comment Resolution Electrical Track`.
- [x] `25_07__ran_3dj_01b_2507` `ran_3dj_01b_2507.pdf` - title is `Update`; first slide suggests `802.3dj D2.0 Comment Resolution Electrical Track`.
- [x] `25_07__ran_3dj_01c_2507` `ran_3dj_01c_2507.pdf` - title is `Update`; first slide suggests `802.3dj D2.0 Comment Resolution Electrical Track`.
- [x] `25_09__issenhuth_3dj_01a_2509` `issenhuth_3dj_01a_2509.pdf` - title is `Update`; first slide suggests `802.3dj D2.1 Comment Resolution Optical Track`.
- [x] `25_09__nicholl_3dj_01a_2509` `nicholl_3dj_01a_2509.pdf` - title is `Update`; first slide suggests `802.3dj D2.1 Comment Resolution Logic Track`.
- [x] `25_09__ran_3dj_01a_2509` `ran_3dj_01a_2509.pdf` - title is `Update`; first slide suggests `802.3dj D2.1 Comment Resolution Electrical Track`.
- [x] `25_09__ran_3dj_01b_2509` `ran_3dj_01b_2509.pdf` - title is `Update`; first slide suggests `802.3dj D2.1 Comment Resolution Electrical Track`.
- [x] `25_09__temprana_3dj_01_2509` `temprana_3dj_01_2509.pdf` - title is `Supporting presentation`; first slide suggests `Supporting presentation for comments 337 & 338`.
- [x] `25_11__brown_3dj_02a_2511` `brown_3dj_02a_2511.pdf` - title is `Update`; first slide suggests `P802.3dj D2.2 Comment Resolution Agenda`.
- [x] `25_11__brown_3dj_02b_2511` `brown_3dj_02b_2511.pdf` - title is `Update`; first slide suggests `P802.3dj D2.2 Comment Resolution Agenda`.
- [x] `25_11__brown_3dj_03a_2511` `brown_3dj_03a_2511.pdf` - title is `Update`; first slide suggests `802.3dj D2.2 Comment Resolution Common Track`.
- [x] `25_11__brown_3dj_03b_2511` `brown_3dj_03b_2511.pdf` - title is `Update`; first slide suggests `802.3dj D2.2 Comment Resolution Common Track`.
- [x] `25_11__issenhuth_3dj_01a_2511` `issenhuth_3dj_01a_2511.pdf` - title is `Update`; first slide suggests `802.3dj D2.2 Comment Resolution Optical Track`.
- [x] `25_11__opsasnick_3dj_01a_2511` `opsasnick_3dj_01a_2511.pdf` - title is `Update (12 Nov)`; first slide suggests `802.3dj D2.2 Comment Resolution Logic Track`.
- [x] `25_11__ran_3dj_01a_2511` `ran_3dj_01a_2511.pdf` - title is `Update`; first slide suggests `802.3dj D2.2 Comment Resolution Electrical Track`.
- [x] `25_11__ran_3dj_01b_2511` `ran_3dj_01b_2511.pdf` - title is `Update`; first slide suggests `802.3dj D2.2 Comment Resolution Electrical Track`.
- [x] `26_01__brown_3dj_02a_2601` `brown_3dj_02a_2601.pdf` - title is `Update`; first slide suggests `P802.3dj D2.3 Comment Resolution Agenda`.
- [x] `26_01__brown_3dj_02b_2601` `brown_3dj_02b_2601.pdf` - title is `Update`; first slide suggests `P802.3dj D2.3 Comment Resolution Agenda`.
- [x] `26_01__opsasnick_3dj_01b_2601` `opsasnick_3dj_01b_2601.pdf` - title is `Update`; first slide suggests `802.3dj D2.3 Comment Resolution Logic Topics`.
- [x] `26_01__ran_3dj_01b_2601` `ran_3dj_01b_2601.pdf` - title is `Update`; first slide suggests `802.3dj D2.3 Comment Resolution Electrical Topics`.
- [x] `26_03__mellitz_3dj_02_2603` `mellitz_3dj_02_2603.pdf` - title is `Supporting Document`; first slide suggests `Document: IEEE802.3dj Modal ERL Proposal`.
- [x] `26_05__brown_3dj_02a_2605` `brown_3dj_02a_2605.pdf` - title is `Update`; first slide suggests `P802.3dj D3.0 Comment Resolution Agenda`.
- [x] `26_05__brown_3dj_02b_2605` `brown_3dj_02b_2605.pdf` - title is `Update`; first slide suggests `P802.3dj D3.0 Comment Resolution Agenda`.
- [x] `26_05__brown_3dj_02c_2605` `brown_3dj_02c_2605.pdf` - title is `Update`; first slide suggests `P802.3dj D3.0 Comment Resolution Agenda`.
- [x] `26_05__brown_3dj_02d_2605` `brown_3dj_02d_2605.pdf` - title is `Update`; first slide suggests `P802.3dj D3.0 Comment Resolution Agenda`.
- [x] `26_05__brown_3dj_02e_2605` `brown_3dj_02e_2605.pdf` - title is `Update`; first slide suggests `P802.3dj D3.0 Comment Resolution Agenda`.
- [x] `26_05__issenhuth_3dj_01a_2605` `issenhuth_3dj_01a_2605.pdf` - title is `Update`; first slide suggests `802.3dj D3.0 Comment Resolution Optical Topics`.
- [x] `26_05__ran_3dj_01a_2605` `ran_3dj_01a_2605.pdf` - title is `Update`; first slide suggests `802.3dj D3.0 Comment Resolution Electrical Topics`.
- [x] `26_05__ran_3dj_01b_2605` `ran_3dj_01b_2605.pdf` - title is `Update`; first slide suggests `802.3dj D3.0 Comment Resolution Electrical Topics`.
- [x] `26_05__ran_3dj_01c_2605` `ran_3dj_01c_2605.pdf` - title is `Update`; first slide suggests `802.3dj D3.0 Comment Resolution Electrical Topics`.

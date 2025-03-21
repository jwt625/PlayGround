
#%%
import csv
import io

csv_text = r'''comment,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Name,,Vainsencher  (APL),Balram (Nat. phot. and PRApp),Jiang (optica),Shao,Forsch,Jiang (this work),Ramp (J Davis),Han (Hong Tang),Peairs (A.N. Cleland),Mirhosseini,Stockill,Jiang (LNSOI),Weaver2023,Yoon2023,Meesala2024,,,,,,,,,,,,,,,,
link1,,https://aip.scitation.org/doi/abs/10.1063/1.4955408,https://www.nature.com/articles/nphoton.2016.46,https://arxiv.org/pdf/1903.00957.pdf,https://arxiv.org/pdf/1907.08593.pdf,https://arxiv.org/pdf/1812.07588.pdf,https://www.nature.com/articles/s41467-020-14863-3,https://aip.scitation.org/doi/full/10.1063/5.0002160,https://www.nature.com/articles/s41467-020-17053-3,https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.061001,https://www.nature.com/articles/s41586-020-3038-6,,https://www.nature.com/articles/s41567-023-02129-w,https://www.nature.com/articles/s41565-023-01515-y,https://doi.org/10.1364/OPTICA.474022,https://www.nature.com/articles/s41567-024-02409-z,,,,,,,,,,,,,,,,
link2,,,,https://opg.optica.org/optica/fulltext.cfm?uri=optica-6-7-845&id=414988,https://opg.optica.org/optica/fulltext.cfm?uri=optica-6-12-1498&id=423626,https://www.nature.com/articles/s41567-019-0673-7,,,,,,https://doi.org/10.1038/s41467-022-34338-x,https://arxiv.org/abs/2210.10739,,,,,,,,,,,,,,,,,,,
Latex Key,,Vainsencher2016,Balram2016,Jiang2019,Shao2019,Forsch2020,Jiang2020,Ramp2020,Han2020,Peairs2020,Mirhosseini2020,Stockill2022,Jiang2023,Weaver2023,Yoon2023,Meesala2024,,,,,,,,,,,,,,,,
time,,2016,2016,2019,2019,2020,2020,2020,2020,2020,2020,2022,2023,2023,2023,2024,,,,,,,,,,,,,,,,
Device approach,,AlN OMC,GaAs OMC,LN OMC,LN Ring,"GaAs, mK",LN POMT,GaAs OMC,AlN disk,AlN-SOI,AlN-SOI,GaP,LN-SOI,LN-SOI,quartz,AlN-SOI,,,,,,,,,,,,,,,,
microwave matching,,no,bad,bad,IDT,bad,IDT,3D cavity,2D resonator,IDT,SC qubit,LC resonator,2D resonator,2D resonator,3D resonator,2D resonator,,,,,,,,,,,,,,,,
g0/2pi,Hz,1.15E+05,1.10E+06,1.20E+05,1.10E+03,1.30E+06,8.00E+04,1.30E+06,1.90E+04,7.34E+05,4.20E+05,7.00E+05,4.20E+05,5.30E+05,5.30E+00,2.70E+05,,,,,,,,,,,,,,,,
g_mu/2pi,Hz,,,,,,,,3.00E+06,1.15E+06,4.50E+06,,4.00E+05,7.40E+06,3.40E+02,8.00E+05,,,,,,,,,,,,,,,,1.05E-34
kappa_i/2pi,Hz,4.70E+09,4.03E+09,5.51E+08,8.08E+07,2.03E+09,4.13E+08,4.30E+09,3.60E+08,3.80E+09,8.00E+08,1.63E+09,5.61E+08,1.34E+09,1.01E+06,6.50E+08,,,,,,,,,,,,,,,,
kappa_e/2pi,Hz,1.05E+10,1.17E+09,2.25E+08,1.43E+07,3.77E+09,8.00E+08,2.30E+09,3.40E+08,6.50E+09,8.10E+08,2.54E+09,5.61E+08,3.65E+09,1.14E+06,6.50E+08,,,,,,,,,,,,,,,,
kappa_tot/2pi,Hz,1.52E+10,5.20E+09,7.76E+08,9.50E+07,5.80E+09,1.21E+09,6.60E+09,7.00E+08,1.03E+10,1.61E+09,4.17E+09,1.12E+09,4.99E+09,2.15E+06,1.30E+09,,,,,,,,,,,,,,,,
eta_o = kappa_e/kappa,,69.08%,22.54%,28.94%,15.00%,65.00%,65.95%,34.85%,48.57%,63.11%,50.31%,60.91%,50.00%,73.15%,53.00%,50.00%,,,,,,,,,,,,,,,,
kappa_mu_e/2pi,Hz,,,,,,,2.05E+06,1.10E+05,3.10E+05,1.00E+07,,3.04E+06,2.20E+06,5.78E+06,1.20E+06,,,,,,,,,,,,,,,,
kappa_mu_i/2pi,Hz,,,,,,,,,,,,,1.23E+06,1.12E+07,5.50E+05,,,,,,,,,,,,,,,,
kappa_mu/2pi,Hz,,,,,,,4.07E+06,4.21E+06,2.46E+07,1.00E+07,,3.10E+06,3.43E+06,1.70E+07,1.75E+06,,,,,,,,,,,,,,,,
gamma_i,Hz,3.50E+06,,5.00E+05,1.06E+06,,1.94E+06,2.90E+06,9.30E+05,6.20E+06,5.00E+05,6.70E+04,3.50E+05,,5.35E+05,1.50E+05,,,,,,,,,,,,,,,,
gamma_e,Hz,1.50E+06,,0.00E+00,2.18E+05,,,,,,,,2.00E+05,,,,,,,,,,,,,,,,,,,
gamma_tot/2pi,Hz,5.00E+06,1.71E+06,5.00E+05,1.28E+06,1.97E+05,1.94E+06,2.90E+06,1.15E+06,6.20E+06,2.75E+06,6.70E+04,5.50E+05,2.63E+06,5.35E+05,1.50E+05,,,,,,,,,,,,,,,,
C0,,6.96E-07,5.43E-04,1.48E-04,3.98E-08,5.92E-03,1.09E-05,3.53E-04,1.79E-06,3.37E-05,1.59E-04,7.02E-03,1.14E-03,8.56E-05,9.77E-11,1.50E-03,,,,,,,,,,,,,,,,
gamma_mu/2pi,,9.55E+02,5.05E-04,8.80E-03,2.18E+05,7.03E-05,1.94E+03,9.15E-06,2.23E+05,2.71E+03,2.25E+06,,2.00E+05,4.10E+07,9.25E-03,1.00E+06,,,,,,,,,,,,,,,,
,,,NA,NA,NA,,,,,,,,,,,,,,,,,,,,,,,,,,,
eta_m,,1.91E-04,2.95E-10,1.76E-08,17.00%,3.57E-10,0.10%,3.16E-12,19.37%,4.37E-04,81.82%,3.60E-06,36.36%,93.97%,1.73E-08,6.69E+00,,,,,,,,,,,,,,,,
Directly measured C0,,,,,,,1.20E-05,,,,,,1.14E-03,,,,,,,,,,,,,,,,,,,
eta0 = 4*C0*eta_o*eta_m,,3.67E-10,1.44E-13,3.02E-12,4.06E-09,5.49E-12,3.17E-08,1.55E-15,6.73E-07,3.72E-08,2.62E-04,6.15E-08,8.32E-04,2.35E-04,3.58E-18,2.00E-02,,,,,,,,,,,,,,,,
eta_total,,9.00E-08,1.00E-10,1.00E-09,1.70E-05,5.50E-12,1.00E-05,3.00E-13,1.00E-03,3.00E-05,1.31E-03,6.80E-08,6.00E-02,9.00E-03,1.20E-08,,,,,,,,,,,,,,,,,
g0/kappa,,7.57E-06,2.12E-04,1.55E-04,1.16E-05,2.24E-04,6.60E-05,1.97E-04,2.71E-05,7.13E-05,2.61E-04,1.68E-04,3.74E-04,1.06E-04,2.47E-06,2.08E-04,,,,,,,,,,,,,,,,
g0^2/kappai/kappa,,1.85E-10,5.78E-08,3.37E-08,1.58E-10,1.44E-07,1.28E-08,5.95E-08,1.43E-09,1.38E-08,1.37E-07,7.21E-08,2.80E-07,4.20E-08,1.29E-11,8.63E-08,,,,,,,,,,,,,,,,
eta fiber to chip,,,,,,0.33,0.65,,,,,,~ 0.5,,,,,,,,,,,,,,,,,,,
added noise,,,,,,,,,,,,,2,6.2,,,,,,,,,,,,,,,,,,
heralding rate,,,,,,,,,,,,,34,,,0.14,,,,,,,,,,,,,,,,
eta internal,,6.54E-03,--,~ 0.01,7.01E-04,7.18E-02,2.57E-02,,1.00E-03,,,,6.00E-02,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
AOM V pi,,,0.7,,0.77,,0.024,,,,,,,,,,,,,,,,,,,,,,,,,
Red or blue,,,,,Blue,,Both,,,,,,,,,,,,,,,,,,,,,,,,,
double check?,,Y,Y,Y,Y,,Y,,,,,,,,,,,,,,,,,,,,,,,,,
measured C0,,,,,,0.006178571429,,,,,,,,,,,,,,,,,,,,,,,,,,
P MW,,,1.00E-07,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Ndot in,,,6.25E+16,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
N phon,,,43,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,For estimating eta_mwwg2mechmode:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
eta GC (assumed),,0.2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
eta O,,6.91E-01,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
C,,3.41E-03,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
omegam,,23750420400,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
nc,,4900,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Ndot in pump,,5.80E+14,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
P pump,,0.00E+00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
P pump actual,,0.00E+00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
eta e->o,,1.96E-08,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
eta o->e,,9.00E-08,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Check 20190828,,Done,Done,Done,Done,Done,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
omegam,,2.38E+10,1.52E+10,1.16E+10,2.06E+10,1.72E+10,1.16E+10,,,,,,,,,,,,,,,,,,,,,,,,,
E bit slow,,5.70E-11,3.03E-08,7.21E-10,2.37E-14,2.52E-08,2.08E-14,,,,,,,,,,,,,,,,,,,,,,,,,
E bit fJ,,5.70E+04,3.03E+07,7.21E+05,2.37E+01,2.52E+07,2.08E+01,,,,,,,,,,,,,,,,,,,,,,,,,
E bit fast,,7.05E-12,1.32E-08,8.11E-09,5.63E-11,1.13E-08,9.69E-14,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
lambda o,,1.52E-06,1.55E-06,1.53E-06,1.57E-06,,1.54E-06,,,,,,,,,,,,,,,,,,,,,,,,,
omegao,,1.24E+15,1.22E+15,1.23E+15,1.20E+15,1.22082E+15,1.22E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,1.21526E+15,,,,,,,,,,,,,,,,
E qubit,,1.33E-06,8.30E-03,1.88E-04,7.81E-09,9.62E-04,3.45E-09,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
E qubit nJ,,1.33E+03,8.30E+06,1.88E+05,7.81E+00,9.62E+05,3.45E+00,4.87E+08,2.37E-10,8.41E+00,5.66E-04,2.02E+02,6.26E-04,1.10E-03,2.69E+08,1.11E-04,,,,,,,,,,,,,,,,
Cooling limited rate (Hz),,7.52E+00,1.20E-03,5.31E-02,1.28E+03,1.04E-02,2.90E+03,2.05E+05,4.23E+04,1.19E+03,1.77E+07,4.96E+01,1.60E+07,9.05E+06,3.72E-05,9.04E+07,,,,,,,,,,,,,,,,
kappa/g0,,1.32E+05,4.72E+03,6.47E+03,8.64E+04,4.46E+03,1.52E+04,,,,,,17328.57143,,,,,,,,,,,,,,,,,,,
2Z0 hbar omegamu,,2.49E-22,1.60E-22,1.21E-22,2.16E-22,1.81E-22,1.22E-22,,,,,,1.22E-22,,,,,,,,,,,,,,,,,,,
gamma^2/4/gammamu,,1.64E+11,3.65E+16,1.78E+14,4.73E+07,3.47E+15,1.22E+10,,,,,,2.77E+09,,,,,,,,,,,,,,,,,,,
V pi,,8.47E-01,1.14E+01,9.52E-01,8.73E-03,3.53E+00,1.85E-02,,,,,,0.01007622066,,,,,,,,,,,,,,,,,,,
V pi old,,2.11E-01,5.33E+00,2.26E+00,3.01E-01,1.67E+00,2.82E-02,,,,,,1.54E-02,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
check E again 20190908,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
hbar omegam,,2.49E-24,1.60E-24,1.21E-24,2.16E-24,1.81E-24,1.22E-24,,,,,,,,,,,,,,,,,,,,,,,,,
kappa/g0,,1.32E+05,4.72E+03,6.47E+03,8.64E+04,4.46E+03,1.52E+04,,,,,,,,,,,,,,,,,,,,,,,,,
omegam/g0,,3.29E+04,2.20E+03,1.53E+04,2.98E+06,2.11E+03,2.64E+04,,,,,,,,,,,,,,,,,,,,,,,,,
E bit slow,,5.70E-11,3.03E-08,7.21E-10,2.37E-14,2.52E-08,2.08E-14,,,,,,,,,,,,,,,,,,,,,,,,,
E bit fast,,,,8.11E-09,5.63E-11,,9.69E-14,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
hbar omegao,,1.30E-19,1.28E-19,1.29E-19,1.26E-19,1.28E-19,1.28E-19,,,,,,,,,,,,,,,,,,,,,,,,,
kappa kappai/g0^2,,5.40E+09,1.73E+07,2.97E+07,6.34E+09,6.97E+06,7.10E+07,,,,,,,,,,,,,,,,,,,,,,,,,
E qubit,,1.33E-06,8.30E-03,1.88E-04,7.81E-09,9.62E-04,3.45E-09,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"20190912, correct E bit",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
slow,,1.43E-11,7.57E-09,,,6.31E-09,,,,,,,,,,,,,,,,,,,,,,,,,,
fast,,,,2.03E-09,1.41E-11,,9.69E-14,,,,,,,,,,,,,,,,,,,,,,,,,'''

# Read the CSV from the text
f = io.StringIO(csv_text)
reader = csv.reader(f)
rows = list(reader)

properties = []
# Loop over each row, starting from row 0
# We'll count non-empty cells from index 2 onward (columns for papers)
for row in rows:
    # Check if row has at least one column (property name)
    if row and row[0].strip():
        # Count non-empty cells in row[2:] (skip first two columns)
        non_empty_count = sum(1 for cell in row[2:] if cell.strip() != "")
        if non_empty_count > 2:
            properties.append((row[0].strip(), non_empty_count))

properties[:50]  # print first 50 for inspection if many exist

# %%
import csv
import json
import re

# Regular expression to check if a string is a valid number (integer or float in scientific notation)
number_re = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')

def convert_cell(cell):
    """
    Process a cell: return None if empty; if the cell is numeric (and does not include a '%'),
    convert it to float; otherwise, return the stripped string.
    """
    cell = cell.strip()
    if cell == "":
        return None
    # If the cell looks like a number (and doesn't include a percentage sign), convert it.
    if "%" not in cell and number_re.match(cell):
        try:
            return float(cell)
        except ValueError:
            return cell
    return cell

def process_csv_to_json(csv_filename, json_filename):
    # Read the CSV file.
    with open(csv_filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    
    # Check that we have at least three columns:
    if not rows or len(rows[0]) < 3:
        print("CSV does not have enough columns to process.")
        return
    
    # The first two columns (index 0 and 1) contain the property name and (sometimes) units.
    # Columns starting at index 2 correspond to individual papers.
    num_papers = len(rows[0]) - 2

    papers = []
    # Process each paper (each column from index 2 onward).
    for col in range(2, 2 + num_papers):
        paper_data = {}
        # Loop over each row to extract properties.
        for row in rows:
            if row and len(row) > 0 and row[0].strip():
                original_key = row[0].strip()
                # Disambiguate duplicate keys by appending a suffix if needed.
                key = original_key
                if key in paper_data:
                    count = 2
                    new_key = f"{original_key} ({count})"
                    while new_key in paper_data:
                        count += 1
                        new_key = f"{original_key} ({count})"
                    key = new_key
                # Get the cell value from the current paper's column.
                value = row[col] if col < len(row) else ""
                paper_data[key] = convert_cell(value)
        # Skip entry if both "Name" and "Latex Key" are missing or empty.
        if not (paper_data.get("Name") or paper_data.get("Latex Key")):
            continue
        papers.append(paper_data)
    
    # Write the JSON output.
    with open(json_filename, "w", encoding="utf-8") as f_out:
        json.dump(papers, f_out, indent=2, ensure_ascii=False)
    print(f"Processed {num_papers} papers and saved to {json_filename}")


#%%
# if __name__ == "__main__":
# Define input and output file names.
csv_filename = "QFC_cleanup.csv"       # Change this to your CSV file path.
json_filename = "output.json"    # Output JSON file.
process_csv_to_json(csv_filename, json_filename)

# %%

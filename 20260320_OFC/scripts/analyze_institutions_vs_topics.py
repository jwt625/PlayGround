#!/usr/bin/env python3
"""
Analyze OFC conference data along two axes:
1. Institutions/Companies (from author affiliations)
2. Technical Topics/Components (from abstracts and titles)

Generate statistics and create a matrix of top 10 institutions vs top 10 topics.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import csv


def load_metadata(json_path):
    """Load the OFC metadata JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_institutions(presentations):
    """
    Extract institution/company names from author affiliations.
    Returns a Counter of institution names.
    """
    institution_counter = Counter()
    
    for pres in presentations:
        authors_json = pres.get('authors_json', '[]')
        if authors_json == '[]':
            continue
            
        try:
            authors = json.loads(authors_json)
            for author in authors:
                affiliation = author.get('affiliation', '').strip()
                if affiliation:
                    # Clean up affiliation
                    affiliation = clean_institution_name(affiliation)
                    if affiliation:
                        institution_counter[affiliation] += 1
        except json.JSONDecodeError:
            continue
    
    return institution_counter


def clean_institution_name(name):
    """Clean and normalize institution names."""
    if not name:
        return ""

    # Remove common suffixes and normalize
    name = name.strip()

    # Remove trailing commas and extra spaces
    name = re.sub(r'\s*,\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)

    # Normalize common variations - comprehensive consolidation
    normalization_map = {
        # Universities
        'Huazhong Univ of Science and Technology': 'Huazhong University of Science and Technology',
        'BUPT': 'Beijing University of Posts and Telecommunications',
        'Beijing University of Posts & Telecom': 'Beijing University of Posts and Telecommunications',
        'UCL': 'University College London',
        'MIT': 'Massachusetts Institute of Technology',
        'Shanghai Jiao Tong university': 'Shanghai Jiao Tong University',
        'Sun Yat-Sen University (CHINA)': 'Sun Yat-Sen University',
        'Cambridge University': 'University of Cambridge',
        'Univ. of Science and Technology of China': 'University of Science and Technology of China',

        # Research Labs
        'Peng Cheng Laboratory': 'Pengcheng Laboratory',
        'Peng Cheng Laboratory (PCL), Shenzhen': 'Pengcheng Laboratory',
        'NICT': 'National Institute of Information and Communication Technology',
        'National Inst of Information & Comm Tech': 'National Institute of Information and Communication Technology',
        'National Institute of Information and Communications Technology': 'National Institute of Information and Communication Technology',

        # Companies - Nokia
        'Nokia Corporation': 'Nokia',
        'Nokia Bell Labs France': 'Nokia Bell Labs',
        'Nokia Solutions and Networks Oy': 'Nokia',
        'Nokia Solutions And Networks Holdings': 'Nokia',
        'Nokia Oy': 'Nokia',

        # Companies - NTT
        'NTT, Inc.': 'NTT',
        'NTT, inc.': 'NTT',
        'NTT Inc': 'NTT',
        'NTT Corporation': 'NTT',
        'Nihon Denshin Denwa Kabushiki Kaisha': 'NTT',
        'Nihon Denshin Denwa Kabushiki Kaisha NTT Device Innovation Center': 'NTT',
        'NTT Access Network Service Systems Laboratories': 'NTT',
        'Device Technology Labs., NTT, Inc.': 'NTT',
        'Kokuritsu Kenkyu Kaihatsu Hojin Joho Tsushin Kenkyu Kiko': 'NTT',

        # Companies - Huawei
        'Huawei Technologies Co., Ltd.': 'Huawei Technologies',
        'Huawei Technologies Co Ltd': 'Huawei Technologies',
        'Huawei Technologies France SAS': 'Huawei Technologies',
        'Huawei Technologies France': 'Huawei Technologies',
        'Huawei Technologies Duesseldorf GmbH': 'Huawei Technologies',
        'Huawei Technologies Deutschland GmbH': 'Huawei Technologies',
        'Huawei Technologies Canada': 'Huawei Technologies',

        # Companies - Samsung
        'Samsung Electronics Co., Ltd.': 'Samsung Electronics',
        'Samsung Electronics Co Ltd': 'Samsung Electronics',

        # Companies - ZTE
        'ZTE corporation': 'ZTE Corporation',
        'ZTE CORPORATION': 'ZTE Corporation',

        # Companies - Microsoft
        'Microsoft Corporation': 'Microsoft',
        'Microsoft Azure Fiber': 'Microsoft',

        # Companies - NVIDIA
        'NVIDIA Corp': 'NVIDIA Corporation',

        # Companies - Cisco
        'Cisco Systems Inc': 'Cisco',

        # Companies - Alibaba
        'Alibaba Cloud Computing': 'Alibaba Cloud',

        # Companies - Nexus Photonics
        'Nexus Photonics, Inc': 'Nexus Photonics',

        # Companies - Lightera
        'Lightera Labs': 'Lightera Laboratories',
        'Lightera laboratories': 'Lightera Laboratories',

        # Companies - HG Genuine Optics
        'HGGenuine Optics Tech Co., Ltd': 'Wuhan HG Genuine Optics Tech Co., Ltd',

        # Companies - Corning
        'Corning Inc.': 'Corning Research and Development Corporation',

        # Companies - GlobalFoundries
        'Globalfoundries Inc Malta': 'GlobalFoundries',

        # Companies - NEC
        'Nihon Denki Kabushiki Kaisha': 'NEC Corporation',

        # Companies - Furukawa
        'Furukawa Denki Kogyo Kabushiki Kaisha': 'Furukawa Electric Co., Ltd.',

        # Companies - Alcatel
        'Alcatel Submarine Network': 'Alcatel Submarine Networks Inc.',

        # Research Centers - Fraunhofer
        'Fraunhofer-Institut fur Nachrichtentechnik Heinrich-Hertz-Institut HHI': 'Fraunhofer HHI',

        # Research Centers - IMEC
        'Interuniversity Microelectronics Center': 'Interuniversitair Micro-Elektronica Centrum',
        'Stichting imec Nederland': 'Interuniversitair Micro-Elektronica Centrum',

        # Research Centers - KIT
        'Karlsruher Institut fur Technologie': 'Karlsruhe Institute of Technology',
        'Karlsruher Institut für Technologie': 'Karlsruhe Institute of Technology',

        # Universities - Stuttgart
        'Universitat Stuttgart': 'Universität Stuttgart',

        # Universities - Chalmers
        'Chalmers tekniska hogskola AB': 'Chalmers University of Technology',

        # Universities - DTU
        'Danmarks Tekniske Universitet': 'DTU Electro',

        # Universities - Ghent
        'Universiteit Gent': 'Ghent University-imec',

        # Universities - CTTC
        'Centre Tecnològic Telecom de Catalunya': 'Centre Tecnologic de Telecomunicacions de Catalunya',
        'CTTC': 'Centre Tecnologic de Telecomunicacions de Catalunya',

        # Universities - Telecom Paris
        'Télécom Paris': 'Telecom Paris',

        # Research - CNIT
        'Consorzio Nazionale Interuniversitario per le Telecomunicazioni': 'CNIT',

        # Research - Xi'an Institute
        "Chinese Academy of Sciences Xi'an Institute of Optics and Precision Mechanics": "Xi'an Institute of Optics and Precision Mechanics",
        "State Key Laboratory of Ultrafast Optical Science and Technology, Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences": "Xi'an Institute of Optics and Precision Mechanics",

        # Research - YOFC
        'Yangtze Optical Fibre & Cable Co': 'Yangtze Optical Fibre and Cable Joint Stock Ltd Co',
        'State Key Laboratory of Optical Fiber and Cable Manufacture Technology, YOFC': 'Yangtze Optical Fibre and Cable Joint Stock Ltd Co',
    }

    # Check for exact matches first
    if name in normalization_map:
        return normalization_map[name]

    # Check for partial matches (case insensitive) for key organizations
    name_lower = name.lower()

    # Special handling for common patterns
    if 'peng cheng lab' in name_lower or 'pengcheng lab' in name_lower:
        return 'Pengcheng Laboratory'
    if 'nokia bell labs' in name_lower:
        return 'Nokia Bell Labs'
    if 'nokia' in name_lower and name != 'Nokia Bell Labs':
        return 'Nokia'
    if 'fraunhofer' in name_lower and 'hhi' in name_lower:
        return 'Fraunhofer HHI'
    if 'imec' in name_lower or ('interuniv' in name_lower and 'micro' in name_lower):
        return 'Interuniversitair Micro-Elektronica Centrum'
    if 'huawei' in name_lower and name != 'Huawei Technologies':
        return 'Huawei Technologies'
    if 'beijing' in name_lower and 'posts' in name_lower and 'telecom' in name_lower:
        return 'Beijing University of Posts and Telecommunications'
    if 'lightera' in name_lower:
        return 'Lightera Laboratories'
    if 'nexus photonics' in name_lower:
        return 'Nexus Photonics'
    if 'nvidia' in name_lower:
        return 'NVIDIA Corporation'
    if 'cisco' in name_lower:
        return 'Cisco'
    if 'zte' in name_lower and 'corporation' not in name_lower:
        return 'ZTE Corporation'

    for key, value in normalization_map.items():
        if key.lower() in name_lower or value.lower() in name_lower:
            return value

    return name


def extract_technical_keywords(presentations):
    """
    Extract technical keywords from abstracts and titles.
    Returns a Counter of technical terms.
    FILTERED to exclude common/mundane/long-lasting basic terms.
    """
    keyword_counter = Counter()

    # Define technical keyword patterns - FILTERED for emerging/distinctive topics only
    # Removed: fiber, laser, bandwidth, data rate, throughput, latency, polarization,
    #          dispersion, nonlinear optics, waveguide, photodetector, coupler,
    #          WDM, EDFA, coherent transmission, DSP, FEC, single-mode fiber,
    #          C-band, L-band, QAM, PAM, PSK
    technical_patterns = [
        # Emerging Transmission & Networking
        (r'\b(SDM|space[- ]division)\b', 'SDM'),
        (r'\b(ROADM|reconfigurable optical)\b', 'ROADM'),
        (r'\b(OTN|optical transport)\b', 'OTN'),

        # Advanced Amplification (not basic EDFA)
        (r'\b(SOA|semiconductor optical amplifier)\b', 'SOA'),
        (r'\b(Raman|raman amplif)\b', 'Raman amplification'),
        (r'\b(optical parametric amplifier|OPA|PPLN)\b', 'OPA/PPLN'),

        # Active Component Research
        (r'\b(modulator|mach[- ]zehnder|MZM|EOM)\b', 'modulator'),
        (r'\b(switch|switching)\b', 'optical switching'),
        (r'\b(transceiver)\b', 'transceiver'),

        # Integration & Platforms (emerging)
        (r'\b(silicon photonics?|SiPh)\b', 'silicon photonics'),
        (r'\b(photonic integrated circuit|PIC)\b', 'PIC'),
        (r'\b(lithium niobate|LiNbO3|LN|LNOI)\b', 'lithium niobate'),
        (r'\b(InP|indium phosphide)\b', 'InP'),
        (r'\b(silicon nitride|SiN)\b', 'silicon nitride'),
        (r'\b(heterogeneous integration)\b', 'heterogeneous integration'),

        # Packaging & Interconnects (HOT TOPICS!)
        (r'\b(CPO|co[- ]packaged|copackaged)\b', 'CPO/co-packaged optics'),
        (r'\b(pluggable)\b', 'pluggable optics'),
        (r'\b(interconnect)\b', 'interconnect'),
        (r'\b(chiplet)\b', 'chiplet'),
        (r'\b(EIC|electronic[- ]photonic)\b', 'EIC'),

        # Photonic Components
        (r'\b(microring|micro[- ]ring|ring resonator)\b', 'microring resonator'),
        (r'\b(AWG|arrayed waveguide)\b', 'AWG'),
        (r'\b(MZI|mach[- ]zehnder interferometer)\b', 'MZI'),

        # Advanced Modulation
        (r'\b(OFDM|orthogonal frequency)\b', 'OFDM'),

        # Advanced Signal Processing
        (r'\b(equalization|equalizer)\b', 'equalization'),
        (r'\b(MIMO)\b', 'MIMO'),

        # AI/ML (emerging in optics)
        (r'\b(machine learning|deep learning|neural network|AI|artificial intelligence)\b', 'ML/AI'),
        (r'\b(CNN|convolutional neural|transformer)\b', 'ML/AI'),
        (r'\b(neuromorphic)\b', 'neuromorphic'),
        (r'\b(photonic computing|optical computing)\b', 'photonic computing'),

        # Emerging Applications
        (r'\b(data[- ]?center|datacenter)\b', 'data center'),
        (r'\b(LiDAR|lidar)\b', 'LiDAR'),
        (r'\b(5G|6G)\b', '5G/6G'),
        (r'\b(quantum|QKD|quantum key)\b', 'quantum'),
        (r'\b(free[- ]space|FSO)\b', 'free-space optical'),
        (r'\b(RoF|radio[- ]over[- ]fiber)\b', 'RoF'),
        (r'\b(sensing|sensor)\b', 'sensing'),

        # Emerging Spectral Bands (not standard C/L)
        (r'\b(S[- ]band)\b', 'S-band'),
        (r'\b(O[- ]band)\b', 'O-band'),
        (r'\b(multi[- ]?band)\b', 'multiband'),

        # Emerging Fiber Technologies
        (r'\b(multi[- ]?mode|MMF)\b', 'multimode fiber'),
        (r'\b(few[- ]?mode|FMF)\b', 'few-mode fiber'),
        (r'\b(multi[- ]?core|MCF)\b', 'multicore fiber'),

        # Cutting Edge Capacity
        (r'\b(petabit|Pb/s)\b', 'petabit-scale'),
        (r'\b(terabit|Tb/s)\b', 'terabit-scale'),

        # Emerging Techniques
        (r'\b(frequency comb|microcomb)\b', 'frequency comb'),
    ]

    for pres in presentations:
        # Combine title and abstract for analysis
        text = f"{pres.get('presentation_title', '')} {pres.get('abstract_text', '')}"

        # Track keywords found in this presentation (use set to avoid double-counting)
        found_keywords = set()

        # Extract keywords using patterns
        for pattern, keyword in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_keywords.add(keyword)

        # Add to counter
        for keyword in found_keywords:
            keyword_counter[keyword] += 1

    return keyword_counter


def build_institution_topic_matrix(presentations, top_institutions, top_keywords):
    """
    Build a matrix showing co-occurrence of institutions and topics.
    Returns:
    - matrix: {institution: {keyword: count}}
    - examples: {institution: {keyword: [presentation_dicts]}}
    """
    matrix = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(lambda: defaultdict(list))

    # Get just the institution and keyword names
    inst_names = [inst for inst, _ in top_institutions]
    kw_names = [kw for kw, _ in top_keywords]

    # Define the same FILTERED technical patterns as in extract_technical_keywords
    technical_patterns = [
        (r'\b(SDM|space[- ]division)\b', 'SDM'),
        (r'\b(ROADM|reconfigurable optical)\b', 'ROADM'),
        (r'\b(OTN|optical transport)\b', 'OTN'),
        (r'\b(SOA|semiconductor optical amplifier)\b', 'SOA'),
        (r'\b(Raman|raman amplif)\b', 'Raman amplification'),
        (r'\b(optical parametric amplifier|OPA|PPLN)\b', 'OPA/PPLN'),
        (r'\b(modulator|mach[- ]zehnder|MZM|EOM)\b', 'modulator'),
        (r'\b(switch|switching)\b', 'optical switching'),
        (r'\b(transceiver)\b', 'transceiver'),
        (r'\b(silicon photonics?|SiPh)\b', 'silicon photonics'),
        (r'\b(photonic integrated circuit|PIC)\b', 'PIC'),
        (r'\b(lithium niobate|LiNbO3|LN|LNOI)\b', 'lithium niobate'),
        (r'\b(InP|indium phosphide)\b', 'InP'),
        (r'\b(silicon nitride|SiN)\b', 'silicon nitride'),
        (r'\b(heterogeneous integration)\b', 'heterogeneous integration'),
        (r'\b(CPO|co[- ]packaged|copackaged)\b', 'CPO/co-packaged optics'),
        (r'\b(pluggable)\b', 'pluggable optics'),
        (r'\b(interconnect)\b', 'interconnect'),
        (r'\b(chiplet)\b', 'chiplet'),
        (r'\b(EIC|electronic[- ]photonic)\b', 'EIC'),
        (r'\b(microring|micro[- ]ring|ring resonator)\b', 'microring resonator'),
        (r'\b(AWG|arrayed waveguide)\b', 'AWG'),
        (r'\b(MZI|mach[- ]zehnder interferometer)\b', 'MZI'),
        (r'\b(OFDM|orthogonal frequency)\b', 'OFDM'),
        (r'\b(equalization|equalizer)\b', 'equalization'),
        (r'\b(MIMO)\b', 'MIMO'),
        (r'\b(machine learning|deep learning|neural network|AI|artificial intelligence)\b', 'ML/AI'),
        (r'\b(CNN|convolutional neural|transformer)\b', 'ML/AI'),
        (r'\b(neuromorphic)\b', 'neuromorphic'),
        (r'\b(photonic computing|optical computing)\b', 'photonic computing'),
        (r'\b(data[- ]?center|datacenter)\b', 'data center'),
        (r'\b(LiDAR|lidar)\b', 'LiDAR'),
        (r'\b(5G|6G)\b', '5G/6G'),
        (r'\b(quantum|QKD|quantum key)\b', 'quantum'),
        (r'\b(free[- ]space|FSO)\b', 'free-space optical'),
        (r'\b(RoF|radio[- ]over[- ]fiber)\b', 'RoF'),
        (r'\b(sensing|sensor)\b', 'sensing'),
        (r'\b(S[- ]band)\b', 'S-band'),
        (r'\b(O[- ]band)\b', 'O-band'),
        (r'\b(multi[- ]?band)\b', 'multiband'),
        (r'\b(multi[- ]?mode|MMF)\b', 'multimode fiber'),
        (r'\b(few[- ]?mode|FMF)\b', 'few-mode fiber'),
        (r'\b(multi[- ]?core|MCF)\b', 'multicore fiber'),
        (r'\b(petabit|Pb/s)\b', 'petabit-scale'),
        (r'\b(terabit|Tb/s)\b', 'terabit-scale'),
        (r'\b(frequency comb|microcomb)\b', 'frequency comb'),
    ]

    for pres in presentations:
        # Get institutions for this presentation
        pres_institutions = set()
        authors_json = pres.get('authors_json', '[]')
        if authors_json != '[]':
            try:
                authors = json.loads(authors_json)
                for author in authors:
                    affiliation = author.get('affiliation', '').strip()
                    if affiliation:
                        affiliation = clean_institution_name(affiliation)
                        if affiliation in inst_names:
                            pres_institutions.add(affiliation)
            except json.JSONDecodeError:
                continue

        # Get keywords for this presentation
        text = f"{pres.get('presentation_title', '')} {pres.get('abstract_text', '')}"

        pres_keywords = set()
        for pattern, keyword in technical_patterns:
            if keyword in kw_names and re.search(pattern, text, re.IGNORECASE):
                pres_keywords.add(keyword)

        # Update matrix and collect examples
        for inst in pres_institutions:
            for kw in pres_keywords:
                matrix[inst][kw] += 1
                # Store presentation info for examples
                examples[inst][kw].append({
                    'code': pres.get('presentation_code', ''),
                    'title': pres.get('presentation_title', ''),
                    'presenter': pres.get('presenter_name', ''),
                    'abstract': pres.get('abstract_text', ''),  # Store full abstract
                })

    return matrix, examples


def print_matrix(matrix, top_institutions, top_keywords, output_dir):
    """Print and save the institution vs topic matrix."""
    inst_names = [inst for inst, _ in top_institutions]
    kw_names = [kw for kw, _ in top_keywords]

    # Print to console
    print("\n" + "=" * 120)
    print("INSTITUTION vs TOPIC MATRIX (Co-occurrence counts)")
    print("=" * 120)

    # Header
    header = "Institution".ljust(45)
    for kw in kw_names:
        header += f"{kw[:12]:>13s}"
    print(header)
    print("-" * 120)

    # Rows
    for inst in inst_names:
        row = inst[:44].ljust(45)
        for kw in kw_names:
            count = matrix[inst][kw]
            row += f"{count:>13d}"
        print(row)

    # Save to CSV
    output_path = output_dir / 'institution_topic_matrix.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Institution'] + kw_names)
        for inst in inst_names:
            row = [inst] + [matrix[inst][kw] for kw in kw_names]
            writer.writerow(row)

    print(f"\nMatrix saved to: {output_path}")


def generate_interactive_html(matrix, examples, top_institutions, top_keywords, output_dir):
    """Generate an interactive HTML heatmap with tooltips showing example talks."""
    import json as json_module

    inst_names = [inst for inst, _ in top_institutions]
    kw_names = [kw for kw, _ in top_keywords]

    # Prepare data for JavaScript
    matrix_data = []
    for i, inst in enumerate(inst_names):
        for j, kw in enumerate(kw_names):
            count = matrix[inst][kw]
            if count > 0:
                # Get top 5 examples
                example_list = examples[inst][kw][:5]
                matrix_data.append({
                    'row': i,
                    'col': j,
                    'institution': inst,
                    'keyword': kw,
                    'count': count,
                    'examples': example_list
                })

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OFC 2026 - Institution vs Topic Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .matrix-container {{
            overflow: auto;
            margin-top: 20px;
            max-height: 80vh;
            border: 2px solid #ddd;
            position: relative;
        }}
        .zoom-controls {{
            position: sticky;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            display: inline-block;
            margin-bottom: 10px;
        }}
        .zoom-btn {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 5px 10px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 3px;
            font-size: 16px;
        }}
        .zoom-btn:hover {{
            background: #1976D2;
        }}
        .filter-controls {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 5px;
        }}
        .filter-btn {{
            padding: 10px 20px;
            margin: 0 5px;
            font-size: 14px;
            cursor: pointer;
            background: #fff;
            color: #333;
            border: 2px solid #4CAF50;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .filter-btn.active {{
            background: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        .filter-btn:hover {{
            background: #66BB6A;
            color: white;
        }}
        table {{
            border-collapse: collapse;
            margin: 0 auto;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            min-width: 80px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th.row-header {{
            background-color: #2196F3;
            text-align: left;
            min-width: 250px;
            max-width: 250px;
            position: sticky;
            left: 0;
            top: 0;
            z-index: 20;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        td.row-header {{
            background-color: #e3f2fd;
            text-align: left;
            font-weight: 500;
            position: sticky;
            left: 0;
            z-index: 9;
            cursor: pointer;
            user-select: none;
            box-shadow: 2px 0 4px rgba(0,0,0,0.1);
        }}
        td.row-header:hover {{
            background-color: #bbdefb;
        }}
        td.row-header.sorted {{
            background-color: #90caf9;
            font-weight: bold;
        }}
        th.clickable {{
            cursor: pointer;
            user-select: none;
        }}
        th.clickable:hover {{
            background-color: #1976D2;
        }}
        th.sorted {{
            background-color: #1565C0;
            font-weight: bold;
        }}
        .cell {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        .cell:hover {{
            background-color: #fff3cd !important;
            transform: scale(1.1);
        }}
        .cell-0 {{ background-color: #f8f9fa; color: #ccc; }}
        .cell-1 {{ background-color: #e8f5e9; }}
        .cell-2 {{ background-color: #c8e6c9; }}
        .cell-3 {{ background-color: #a5d6a7; }}
        .cell-4 {{ background-color: #81c784; }}
        .cell-5 {{ background-color: #66bb6a; }}
        .cell-high {{ background-color: #43a047; color: white; }}

        .tooltip {{
            position: fixed;
            background: white;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            padding-bottom: 40px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            display: none;
            pointer-events: auto;
        }}
        .tooltip.pinned {{
            border-color: #4CAF50;
            border-width: 3px;
        }}
        .tooltip-close {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            cursor: pointer;
            font-weight: bold;
            display: none;
        }}
        .tooltip.pinned .tooltip-close {{
            display: block;
        }}
        .tooltip h3 {{
            margin: 0 0 10px 0;
            color: #2196F3;
            font-size: 16px;
        }}
        .tooltip .count {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }}
        .tooltip .example {{
            margin: 10px 0;
            padding: 10px;
            background: #f5f5f5;
            border-left: 3px solid #4CAF50;
            border-radius: 4px;
        }}
        .tooltip .example-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .tooltip .example-code {{
            color: #2196F3;
            font-size: 12px;
        }}
        .tooltip .example-presenter {{
            color: #666;
            font-size: 12px;
            font-style: italic;
        }}
        .tooltip .example-abstract {{
            color: #555;
            font-size: 11px;
            margin-top: 5px;
            line-height: 1.4;
            cursor: pointer;
            position: relative;
        }}
        .tooltip .example-abstract.collapsed {{
            max-height: 3.6em;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .tooltip .example-abstract.collapsed::after {{
            content: ' ... (click to expand)';
            color: #2196F3;
            font-weight: bold;
        }}
        .tooltip .example-abstract.expanded::after {{
            content: ' (click to collapse)';
            color: #2196F3;
            font-weight: bold;
            display: block;
            margin-top: 5px;
        }}
        .legend {{
            margin: 20px 0;
            text-align: center;
        }}
        .legend-item {{
            display: inline-block;
            margin: 0 10px;
        }}
        .legend-box {{
            display: inline-block;
            width: 30px;
            height: 20px;
            border: 1px solid #ddd;
            vertical-align: middle;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OFC 2026 Conference Analysis</h1>
        <h2 style="text-align: center; color: #666;">Institutions × Emerging Technical Topics (Expanded Coverage)</h2>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Top 50 institutions + top 10 institutions per topic | Top 50 topics + top 10 topics per institution
        </p>

        <div class="zoom-controls">
            <button class="zoom-btn" onclick="zoomIn()">+</button>
            <button class="zoom-btn" onclick="zoomOut()">−</button>
            <button class="zoom-btn" onclick="resetZoom()">Reset</button>
            <span style="margin-left: 10px;">Zoom: <span id="zoom-level">100</span>%</span>
        </div>

        <p style="text-align: center; color: #666; font-size: 0.85em; margin-top: 10px;">
            💡 <strong>Tip:</strong> Click column headers to sort institutions by topic activity | Click row headers to sort topics by institution activity
        </p>

        <div class="filter-controls">
            <strong>Filter by Institution Type:</strong>
            <button class="filter-btn active" onclick="filterInstitutions('all')">All</button>
            <button class="filter-btn" onclick="filterInstitutions('academic')">Universities & Labs</button>
            <button class="filter-btn" onclick="filterInstitutions('industry')">Companies</button>
        </div>

        <div class="legend">
            <div class="legend-item"><span class="legend-box" style="background-color: #f8f9fa;"></span>0</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #e8f5e9;"></span>1</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #c8e6c9;"></span>2</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #a5d6a7;"></span>3</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #81c784;"></span>4</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #66bb6a;"></span>5</div>
            <div class="legend-item"><span class="legend-box" style="background-color: #43a047;"></span>6+</div>
        </div>

        <div class="matrix-container">
            <table id="matrix">
                <thead>
                    <tr>
                        <th class="row-header">Institution</th>
"""

    # Add column headers (clickable for sorting)
    for j, kw in enumerate(kw_names):
        html_content += f"                        <th class='clickable' data-col='{j}'>{kw}</th>\n"

    html_content += """                    </tr>
                </thead>
                <tbody>
"""

    # Add data rows
    for i, inst in enumerate(inst_names):
        html_content += f"                    <tr data-row='{i}'>\n"
        html_content += f"                        <td class='row-header' data-row='{i}'>{inst}</td>\n"
        for j, kw in enumerate(kw_names):
            count = matrix[inst][kw]
            cell_class = 'cell-0' if count == 0 else f'cell-{min(count, 5)}' if count <= 5 else 'cell-high'
            html_content += f"                        <td class='cell {cell_class}' data-row='{i}' data-col='{j}'>{count if count > 0 else ''}</td>\n"
        html_content += "                    </tr>\n"

    html_content += """                </tbody>
            </table>
        </div>
    </div>

    <div class="tooltip" id="tooltip">
        <button class="tooltip-close" onclick="closeTooltip()">×</button>
        <div id="tooltip-content"></div>
    </div>

    <script>
        const matrixData = """ + json_module.dumps(matrix_data, indent=8) + """;

        const tooltip = document.getElementById('tooltip');
        const tooltipContent = document.getElementById('tooltip-content');
        const cells = document.querySelectorAll('.cell');
        const matrixTable = document.getElementById('matrix');

        let isPinned = false;
        let currentZoom = 1.0;

        // Create lookup map for quick access
        const dataMap = {};
        matrixData.forEach(item => {
            const key = `${item.row}-${item.col}`;
            dataMap[key] = item;
        });

        function closeTooltip() {
            tooltip.style.display = 'none';
            tooltip.classList.remove('pinned');
            isPinned = false;
        }

        function showTooltip(data, x, y) {
            let html = `<h3>${data.institution} × ${data.keyword}</h3>`;
            html += `<div class="count">Co-occurrences: ${data.count}</div>`;

            if (data.examples && data.examples.length > 0) {
                html += `<div style="margin-top: 10px; font-weight: bold;">Example Presentations:</div>`;
                data.examples.forEach((ex, idx) => {
                    html += `<div class="example">`;
                    html += `<div class="example-code">${ex.code}</div>`;
                    html += `<div class="example-title">${ex.title}</div>`;
                    if (ex.presenter) {
                        html += `<div class="example-presenter">Presenter: ${ex.presenter}</div>`;
                    }
                    if (ex.abstract) {
                        html += `<div class="example-abstract collapsed" data-idx="${idx}">${ex.abstract}</div>`;
                    }
                    html += `</div>`;
                });
            }

            tooltipContent.innerHTML = html;
            tooltip.style.display = 'block';

            // Position tooltip
            let left = x + 15;
            let top = y + 15;

            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';

            // Check if tooltip goes off right edge
            setTimeout(() => {
                const tooltipRect = tooltip.getBoundingClientRect();
                if (tooltipRect.right > window.innerWidth) {
                    tooltip.style.left = (x - tooltipRect.width - 15) + 'px';
                }
                if (tooltipRect.bottom > window.innerHeight) {
                    tooltip.style.top = (y - tooltipRect.height - 15) + 'px';
                }
            }, 0);

            // Add click handlers for abstract expansion
            tooltip.querySelectorAll('.example-abstract').forEach(abstract => {
                abstract.addEventListener('click', function(e) {
                    e.stopPropagation();
                    this.classList.toggle('collapsed');
                    this.classList.toggle('expanded');
                });
            });
        }

        // Function to attach cell event listeners
        function attachCellListeners() {
            const cells = document.querySelectorAll('td.cell');
            cells.forEach(cell => {
                cell.addEventListener('mouseenter', function(e) {
                    if (isPinned) return;

                    const row = this.getAttribute('data-row');
                    const col = this.getAttribute('data-col');
                    const key = `${row}-${col}`;
                    const data = dataMap[key];

                    if (data && data.count > 0) {
                        showTooltip(data, e.clientX, e.clientY);
                    }
                });

                cell.addEventListener('mouseleave', function() {
                    if (!isPinned) {
                        tooltip.style.display = 'none';
                    }
                });

                cell.addEventListener('click', function(e) {
                    const row = this.getAttribute('data-row');
                    const col = this.getAttribute('data-col');
                    const key = `${row}-${col}`;
                    const data = dataMap[key];

                    if (data && data.count > 0) {
                        isPinned = true;
                        tooltip.classList.add('pinned');
                        showTooltip(data, e.clientX, e.clientY);
                        e.stopPropagation();
                    }
                });

                cell.addEventListener('mousemove', function(e) {
                    if (!isPinned && tooltip.style.display === 'block') {
                        let left = e.clientX + 15;
                        let top = e.clientY + 15;
                        tooltip.style.left = left + 'px';
                        tooltip.style.top = top + 'px';
                    }
                });
            });
        }

        // Initial attachment of cell listeners
        attachCellListeners();

        // Close tooltip when clicking outside
        document.addEventListener('click', function(e) {
            if (isPinned && !tooltip.contains(e.target)) {
                closeTooltip();
            }
        });

        // Sorting functionality
        let currentSortCol = null;
        let currentSortRow = null;

        // Store original order for reset - deep clone to preserve original state
        const originalRowOrder = Array.from(document.querySelectorAll('tbody tr')).map(row => row.cloneNode(true));
        const originalColOrder = Array.from(document.querySelectorAll('thead th')).slice(1).map(col => col.cloneNode(true)); // Skip row header

        // Function to attach header click listeners
        function attachHeaderListeners() {
            // Sort by column (topic) - reorder rows by activity in that topic
            document.querySelectorAll('th.clickable').forEach(header => {
                header.addEventListener('click', function() {
                    const col = parseInt(this.getAttribute('data-col'));

                    // Clear row sorting
                    document.querySelectorAll('td.row-header').forEach(h => h.classList.remove('sorted'));
                    currentSortRow = null;

                    // Toggle column sorting
                    const wasSorted = this.classList.contains('sorted');
                    document.querySelectorAll('th.clickable').forEach(h => h.classList.remove('sorted'));

                    if (!wasSorted) {
                        this.classList.add('sorted');
                        currentSortCol = col;
                        sortByColumn(col);
                    } else {
                        currentSortCol = null;
                        resetOrder();
                    }
                });
            });

            // Sort by row (institution) - reorder columns by that institution's activity
            document.querySelectorAll('td.row-header').forEach(header => {
                header.addEventListener('click', function() {
                    const row = parseInt(this.getAttribute('data-row'));

                    // Clear column sorting
                    document.querySelectorAll('th.clickable').forEach(h => h.classList.remove('sorted'));
                    currentSortCol = null;

                    // Toggle row sorting
                    const wasSorted = this.classList.contains('sorted');
                    document.querySelectorAll('td.row-header').forEach(h => h.classList.remove('sorted'));

                    if (!wasSorted) {
                        this.classList.add('sorted');
                        currentSortRow = row;
                        sortByRow(row);
                    } else {
                        currentSortRow = null;
                        resetOrder();
                    }
                });
            });
        }

        // Initial attachment of header listeners
        attachHeaderListeners();

        function sortByColumn(colIndex) {
            const tbody = document.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Sort rows by the value in the specified column (descending)
            rows.sort((a, b) => {
                const cellA = a.querySelectorAll('td')[colIndex + 1]; // +1 for row header
                const cellB = b.querySelectorAll('td')[colIndex + 1];
                const valA = parseInt(cellA.textContent.trim()) || 0;
                const valB = parseInt(cellB.textContent.trim()) || 0;
                return valB - valA; // Descending
            });

            // Reorder rows in DOM
            rows.forEach(row => tbody.appendChild(row));
        }

        function sortByRow(rowIndex) {
            const table = document.querySelector('table');
            const headerRow = table.querySelector('thead tr');
            const tbody = table.querySelector('tbody');
            const targetRow = tbody.querySelectorAll('tr')[rowIndex];

            // Get all column headers (excluding first row header)
            const headers = Array.from(headerRow.querySelectorAll('th')).slice(1);

            // Get values from the target row
            const cells = Array.from(targetRow.querySelectorAll('td')).slice(1); // Skip row header

            // Create array of [colIndex, value] pairs
            const colValues = cells.map((cell, idx) => ({
                index: idx,
                value: parseInt(cell.textContent.trim()) || 0
            }));

            // Sort by value (descending)
            colValues.sort((a, b) => b.value - a.value);

            // Create new column order
            const newOrder = colValues.map(cv => cv.index);

            // Reorder columns in header
            const newHeaders = [headerRow.querySelector('th.row-header')];
            newOrder.forEach(idx => newHeaders.push(headers[idx]));

            // Clear and rebuild header row
            headerRow.innerHTML = '';
            newHeaders.forEach(h => headerRow.appendChild(h));

            // Reorder columns in all data rows
            tbody.querySelectorAll('tr').forEach(row => {
                const rowCells = Array.from(row.querySelectorAll('td'));
                const rowHeader = rowCells[0];
                const dataCells = rowCells.slice(1);

                const newCells = [rowHeader];
                newOrder.forEach(idx => newCells.push(dataCells[idx]));

                row.innerHTML = '';
                newCells.forEach(c => row.appendChild(c));
            });
        }

        function resetOrder() {
            // Restore original row order
            const tbody = document.querySelector('tbody');
            tbody.innerHTML = '';
            // Clone from our stored deep copies
            originalRowOrder.forEach(row => tbody.appendChild(row.cloneNode(true)));

            // Restore original column order
            const table = document.querySelector('table');
            const headerRow = table.querySelector('thead tr');
            const rowHeader = headerRow.querySelector('th.row-header');

            // Rebuild header
            headerRow.innerHTML = '';
            headerRow.appendChild(rowHeader);
            originalColOrder.forEach(col => headerRow.appendChild(col.cloneNode(true)));

            // Re-attach event listeners
            attachCellListeners();
            attachHeaderListeners();
        }

        // Zoom functions
        function zoomIn() {
            currentZoom = Math.min(currentZoom + 0.2, 3.0);
            applyZoom();
        }

        function zoomOut() {
            currentZoom = Math.max(currentZoom - 0.2, 0.5);
            applyZoom();
        }

        function resetZoom() {
            currentZoom = 1.0;
            applyZoom();
        }

        function applyZoom() {
            matrixTable.style.transform = `scale(${currentZoom})`;
            matrixTable.style.transformOrigin = 'top left';
            document.getElementById('zoom-level').textContent = Math.round(currentZoom * 100);
        }

        // Institution type filtering
        function isAcademic(institutionName) {
            const name = institutionName.toLowerCase();
            return name.includes('universit') ||
                   name.includes('lab') ||
                   name.includes('college') ||
                   name.includes('institut') ||
                   name.includes('school') ||
                   name.includes('academy');
        }

        function filterInstitutions(type) {
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Get all data rows
            const rows = document.querySelectorAll('tbody tr');

            rows.forEach(row => {
                const institutionCell = row.querySelector('td.row-header');
                const institutionName = institutionCell.textContent.trim();
                const academic = isAcademic(institutionName);

                let shouldShow = false;
                if (type === 'all') {
                    shouldShow = true;
                } else if (type === 'academic' && academic) {
                    shouldShow = true;
                } else if (type === 'industry' && !academic) {
                    shouldShow = true;
                }

                // Use visibility: collapse for proper table rendering
                if (shouldShow) {
                    row.style.visibility = '';
                    row.style.display = '';
                } else {
                    row.style.visibility = 'collapse';
                }
            });
        }
    </script>
</body>
</html>
"""

    # Save HTML file
    html_path = output_dir / 'interactive_matrix.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nInteractive HTML matrix saved to: {html_path}")
    return html_path


def save_detailed_stats(institutions, keywords, top_institutions, top_keywords, output_dir):
    """Save detailed statistics to CSV files."""
    # Save all institutions
    inst_path = output_dir / 'all_institutions.csv'
    with open(inst_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Institution', 'Count'])
        for i, (inst, count) in enumerate(institutions.most_common(), 1):
            writer.writerow([i, inst, count])
    print(f"All institutions saved to: {inst_path}")

    # Save all keywords
    kw_path = output_dir / 'all_keywords.csv'
    with open(kw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Keyword', 'Count'])
        for i, (kw, count) in enumerate(keywords.most_common(), 1):
            writer.writerow([i, kw, count])
    print(f"All keywords saved to: {kw_path}")


def main():
    # Load metadata
    metadata_path = Path('output/full_metadata/ofc_full_metadata.json')
    data = load_metadata(metadata_path)

    # Filter to presentations only
    presentations = [r for r in data if r['record_kind'] == 'presentation']

    print(f"Total presentations: {len(presentations)}\n")

    # Extract institutions
    print("Extracting institutions from author affiliations...")
    institutions = extract_institutions(presentations)

    print(f"Total unique institutions: {len(institutions)}")
    print(f"Total institution mentions: {sum(institutions.values())}\n")

    # Extract technical keywords
    print("Extracting technical keywords from abstracts and titles...")
    keywords = extract_technical_keywords(presentations)

    print(f"Total unique keywords: {len(keywords)}")
    print(f"Total keyword mentions: {sum(keywords.values())}\n")

    # Get top 50 of each (changed from 10)
    top_institutions = institutions.most_common(50)
    top_keywords = keywords.most_common(50)

    # Expand the lists to ensure comprehensive coverage:
    # 1. For each top topic, include top 10 institutions working on it
    # 2. For each top institution, include top 10 topics they work on

    print("\nExpanding institution and topic lists for comprehensive coverage...")

    expanded_institutions = set(inst for inst, _ in top_institutions)
    expanded_keywords = set(kw for kw, _ in top_keywords)

    # For each top keyword, find top 10 institutions
    for keyword, _ in top_keywords:
        # Count institutions for this keyword
        inst_for_kw = Counter()
        for pres in presentations:
            kws = extract_technical_keywords([pres])
            if keyword in kws:
                insts = extract_institutions([pres])
                for inst in insts:
                    inst_for_kw[inst] += 1

        # Add top 10 institutions for this keyword
        for inst, _ in inst_for_kw.most_common(10):
            expanded_institutions.add(inst)

    # For each top institution, find top 10 keywords
    for institution, _ in top_institutions:
        # Count keywords for this institution
        kw_for_inst = Counter()
        for pres in presentations:
            insts = extract_institutions([pres])
            if institution in insts:
                kws = extract_technical_keywords([pres])
                for kw in kws:
                    kw_for_inst[kw] += 1

        # Add top 10 keywords for this institution
        for kw, _ in kw_for_inst.most_common(10):
            expanded_keywords.add(kw)

    # Rebuild the lists with expanded coverage, sorted by original counts
    expanded_inst_list = [(inst, institutions[inst]) for inst in expanded_institutions]
    expanded_inst_list.sort(key=lambda x: x[1], reverse=True)

    expanded_kw_list = [(kw, keywords[kw]) for kw in expanded_keywords]
    expanded_kw_list.sort(key=lambda x: x[1], reverse=True)

    print(f"Expanded from {len(top_institutions)} to {len(expanded_inst_list)} institutions")
    print(f"Expanded from {len(top_keywords)} to {len(expanded_kw_list)} keywords")

    # Use expanded lists
    top_institutions = expanded_inst_list
    top_keywords = expanded_kw_list

    print("=" * 80)
    print(f"TOP {len(top_institutions)} INSTITUTIONS/COMPANIES (expanded)")
    print("=" * 80)
    for i, (inst, count) in enumerate(top_institutions[:20], 1):  # Show first 20 in console
        print(f"{i:2d}. {inst:60s} {count:4d}")
    print(f"... and {len(top_institutions) - 20} more")

    print("\n" + "=" * 80)
    print(f"TOP {len(top_keywords)} TECHNICAL KEYWORDS/TOPICS (expanded)")
    print("=" * 80)
    for i, (kw, count) in enumerate(top_keywords[:20], 1):  # Show first 20 in console
        print(f"{i:2d}. {kw:60s} {count:4d}")
    print(f"... and {len(top_keywords) - 20} more")

    # Build and display matrix
    print("\nBuilding institution vs topic matrix...")
    matrix, examples = build_institution_topic_matrix(presentations, top_institutions, top_keywords)

    # Create output directory
    output_dir = Path('output/institution_topic_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print and save matrix
    print_matrix(matrix, top_institutions, top_keywords, output_dir)

    # Generate interactive HTML
    html_path = generate_interactive_html(matrix, examples, top_institutions, top_keywords, output_dir)

    # Save detailed statistics
    save_detailed_stats(institutions, keywords, top_institutions, top_keywords, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Open the interactive matrix: {html_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()


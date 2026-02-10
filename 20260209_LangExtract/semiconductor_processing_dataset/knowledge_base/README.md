# Knowledge Base

This directory contains normalized databases and ontologies built from extracted data.

## Files

All knowledge base files use this top-level shape:
```json
{
  "_metadata": {
    "version": "0.1.0",
    "last_updated": "YYYY-MM-DD"
  },
  "<collection_key>": {}
}
```

Collection keys by file:
- `chemical_database.json` -> `chemicals`
- `equipment_database.json` -> `equipment`
- `abbreviations.json` -> `abbreviations`
- `process_ontology.json` -> `ontology`

### chemical_database.json
Normalized chemical names and their variants.

Purpose:
- Map different representations to canonical forms
- Handle subscripts, superscripts, abbreviations
- Track chemical roles (precursor, etchant, resist, etc.)

Example structure:
```json
{
  "_metadata": {
    "version": "0.1.0",
    "last_updated": "2026-02-10"
  },
  "chemicals": {
    "SiO2": {
      "canonical_name": "Silicon dioxide",
      "formula": "SiO2",
      "variants": [
        "SiO₂",
        "silicon oxide",
        "silica",
        "thermal oxide",
        "TEOS oxide"
      ],
      "cas_number": "7631-86-9",
      "common_roles": ["dielectric", "mask", "cladding"],
      "typical_processes": ["thermal_oxidation", "PECVD", "sputtering"]
    },
    "HF": {
      "canonical_name": "Hydrofluoric acid",
      "formula": "HF",
      "variants": [
        "hydrofluoric acid",
        "HF solution",
        "buffered HF",
        "BOE"
      ],
      "cas_number": "7664-39-3",
      "common_concentrations": ["49%", "1%", "7:1 BOE"],
      "common_roles": ["etchant"],
      "etches": ["SiO2", "native oxide"],
      "safety_notes": "Highly toxic, requires special handling"
    }
  }
}
```

### equipment_database.json
Tool specifications and standardized names.

Purpose:
- Normalize equipment names across papers
- Track vendor, model, capabilities
- Map generic names to specific tools

Example structure:
```json
{
  "_metadata": {
    "version": "0.1.0",
    "last_updated": "2026-02-10"
  },
  "equipment": {
    "oxford_plasmalab_100": {
      "canonical_name": "Oxford Plasmalab 100",
      "vendor": "Oxford Instruments",
      "type": "PECVD",
      "variants": [
        "Plasmalab 100",
        "Oxford PECVD",
        "Oxford Plasmalab"
      ],
      "capabilities": ["PECVD", "RIE", "ICP"],
      "typical_materials": ["SiO2", "SiNx", "a-Si"],
      "common_in_labs": ["Yale", "MIT", "Berkeley"]
    },
    "ebl_generic": {
      "canonical_name": "Electron beam lithography system",
      "type": "lithography",
      "variants": [
        "EBL",
        "e-beam lithography",
        "electron beam writer"
      ],
      "specific_models": [
        "Raith EBPG 5000+",
        "Elionix ELS-F125",
        "JEOL JBX-6300FS",
        "Heidelberg µPG 101"
      ]
    }
  }
}
```

### abbreviations.json
Domain-specific abbreviations and expansions.

Purpose:
- Expand abbreviations consistently
- Handle context-dependent meanings
- Track domain-specific terminology

Example structure:
```json
{
  "_metadata": {
    "version": "0.1.0",
    "last_updated": "2026-02-10"
  },
  "abbreviations": {
    "PECVD": {
      "expansion": "Plasma-Enhanced Chemical Vapor Deposition",
      "domain": "general",
      "related_terms": ["CVD", "LPCVD", "APCVD"]
    },
    "BOE": {
      "expansion": "Buffered Oxide Etch",
      "domain": "wet_etching",
      "typical_composition": "HF:NH4F (1:6 or 1:7)",
      "also_known_as": ["BHF", "buffered HF"]
    },
    "ICP-RIE": {
      "expansion": "Inductively Coupled Plasma Reactive Ion Etching",
      "domain": "dry_etching",
      "related_terms": ["RIE", "DRIE", "ICP"]
    },
    "RT": {
      "expansion": "Room Temperature",
      "domain": "general",
      "typical_range": "20-25°C",
      "context_dependent": true
    },
    "PMMA": {
      "expansion": "Poly(methyl methacrylate)",
      "domain": "lithography",
      "common_uses": ["e-beam resist", "transfer layer"],
      "typical_grades": ["950K A4", "495K A2", "950K A2"]
    }
  }
}
```

### process_ontology.json
Hierarchical taxonomy of process steps.

Purpose:
- Categorize process types
- Define relationships between processes
- Enable semantic search and analysis

Example structure:
```json
{
  "_metadata": {
    "version": "0.1.0",
    "last_updated": "2026-02-10"
  },
  "ontology": {
    "deposition": {
      "description": "Material deposition processes",
      "subcategories": {
        "physical_vapor_deposition": {
          "description": "PVD processes",
          "methods": [
            "evaporation",
            "sputtering",
            "molecular_beam_epitaxy"
          ],
          "evaporation": {
            "types": ["thermal", "e-beam"],
            "typical_materials": ["Al", "Au", "Cr", "Ti"],
            "typical_parameters": ["pressure", "rate", "temperature"]
          },
          "sputtering": {
            "types": ["DC", "RF", "magnetron"],
            "typical_materials": ["Al", "Nb", "Ta", "SiO2"],
            "typical_parameters": ["power", "pressure", "gas_flow"]
          }
        },
        "chemical_vapor_deposition": {
          "description": "CVD processes",
          "methods": [
            "LPCVD",
            "PECVD",
            "APCVD",
            "ALD"
          ],
          "PECVD": {
            "typical_materials": ["SiO2", "SiNx", "a-Si"],
            "typical_parameters": ["temperature", "pressure", "power", "gas_flows"]
          }
        }
      }
    },
    "etching": {
      "description": "Material removal processes",
      "subcategories": {
        "wet_etching": {
          "description": "Liquid-phase etching",
          "types": ["isotropic", "anisotropic"],
          "typical_chemicals": ["HF", "BOE", "KOH", "TMAH", "piranha"]
        },
        "dry_etching": {
          "description": "Plasma-based etching",
          "methods": ["RIE", "ICP-RIE", "DRIE", "ion_milling"],
          "typical_parameters": ["power", "pressure", "gas_composition", "bias"]
        }
      }
    },
    "lithography": {
      "description": "Pattern definition processes",
      "methods": [
        "photolithography",
        "electron_beam_lithography",
        "nanoimprint"
      ],
      "electron_beam_lithography": {
        "typical_resists": ["PMMA", "HSQ", "ZEP"],
        "typical_parameters": ["dose", "beam_current", "acceleration_voltage"]
      }
    },
    "thermal_treatment": {
      "description": "Temperature-based processes",
      "methods": [
        "annealing",
        "oxidation",
        "diffusion"
      ],
      "annealing": {
        "purposes": ["stress_relief", "crystallization", "dopant_activation"],
        "atmospheres": ["vacuum", "forming_gas", "N2", "O2"]
      }
    }
  }
}
```

## Building the Knowledge Base

### Phase 1: Initial Population
- Extract from high-quality theses and review papers
- Manual curation of common chemicals and equipment
- Define core process taxonomy

### Phase 2: Automated Expansion
- Run LangExtract on document corpus
- Aggregate extracted entities
- Identify variants through clustering

### Phase 3: Normalization
- Map variants to canonical forms
- Resolve conflicts and ambiguities
- Expert validation

### Phase 4: Continuous Update
- Add new entities from new documents
- Refine mappings based on usage
- Track temporal changes (e.g., new equipment models)

## Usage

These databases are used for:
- Normalizing LangExtract outputs
- Validating extracted entities
- Semantic search across documents
- Process comparison and analysis
- Recipe standardization

## Maintenance

- Version control all JSON files
- Document changes and rationale
- Regular expert review
- Community contributions welcome

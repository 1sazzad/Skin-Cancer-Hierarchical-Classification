# Phase 10B HIBA Official Metadata Acquisition Protocol

## Authorization boundary

Phase 10B prepares tooling for official metadata acquisition. It does not
authorize a network request by itself. A human must review this protocol before
running the acquisition command with `--authorize-network-acquisition`.

This phase does not authorize image download, the Phase 10A real audit,
external-evaluation manifest creation, cohort selection, label mapping,
training, checkpoint access, Azure use, inference, or performance inspection.
Image acquisition remains a separate later gate.

HIBA remains a mandatory frozen zero-shot external-evaluation candidate. No
HIBA metadata or later result may influence model development, preprocessing,
thresholds, calibration, checkpoint selection, or architecture selection.

## Official release lock

The only permitted source is ISIC Archive API v2 for:

- collection ID: `251`
- DOI: `10.34970/587329`
- title: `Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)`
- expected images: `1616`
- expected specified lesions: `1246`
- expected specified patients: `623`

Collection `175` and DOI `10.34970/559884` are explicitly prohibited
substitutes. The release contains both clinical and dermoscopic images; Phase
10B inventories modality values exactly and does not select a cohort.

## Directory and artifact policy

All acquired material remains ignored beneath `data/external/hiba/`:

```text
data/external/hiba/
    source/
    metadata/
    images/
```

The metadata acquisition tool plans these finalized artifacts:

- `source/collection_251.json`
- `source/collection_251_attribution.json`
- `source/acquisition_environment.json`
- `source/acquisition_request_log.json`
- `metadata/collection_251_images.raw.jsonl`

The offline inventory tool subsequently plans:

- `metadata/collection_251_metadata_inventory.json`
- `metadata/collection_251_metadata_inventory.csv`

Both tools refuse existing finalized outputs, constrain paths beneath the HIBA
root, serialize to temporary sibling files, publish only after all
serialization succeeds, remove temporary files after failure, and roll back
partially published new outputs.

## Acquisition sequence

1. Fetch collection 251 metadata from the structured API v2 endpoint.
2. Verify collection ID, DOI, exact title, and declared image count before
   publishing any final artifact.
3. Page through the structured collection-image search until `next` is null.
4. Preserve each returned image object as one JSON object in raw JSONL without
   adding, removing, mapping, or inferring metadata values.
5. Reject empty or duplicate image IDs and require exactly 1616 records.
6. Record endpoint URL, UTC timestamp, HTTP status, page number, and item count.
7. Publish the complete source bundle transactionally.

Requests use a bounded retry count, exponential backoff, timeout, and explicit
research-project user agent. Live access is refused unless
`--authorize-network-acquisition` is supplied. Fixture mode is offline and
mutually exclusive with network authorization. No HTML scraping is permitted.

## Offline metadata inventory

The inventory script reads only the preserved raw JSONL and never calls the
network. It reports exact values, missingness, and nested source paths for:

- image, patient, and lesion identifiers;
- diagnosis and diagnosis-confirmation fields;
- modality;
- licence and attribution;
- file extension or MIME metadata where available;
- public/private status where available.

Missing fields stay missing. Unknown values remain unchanged. Diagnosis
substring inference, benign/malignant classification, label mapping,
dermoscopic cohort selection, evaluation approval, and data splitting are
forbidden. Exactly 1616 unique image records are required.

Inventory status remains
`metadata_inventory_pending_human_review`. The Phase 10A label mapping,
including unresolved benign vocabulary, cannot be amended until the exact
official metadata inventory has been reviewed and a prospective human decision
is recorded before inference.

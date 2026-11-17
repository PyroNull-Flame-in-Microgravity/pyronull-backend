# PSI-107 / PSI-69 Model Artifact Provenance and Redistribution Checklist

**Public redistribution status: NOT CLEARED.** Do not add the three model files below to a public Git commit, public GitHub release, public container image, or public artifact registry until written redistribution permission is documented. The local `.gitignore` rules are intended to prevent accidental `git add`.

## Artifact inventory observed in this checkout

SHA-256 values were computed from the current files. These values identify the present bytes; they do not establish provenance, ownership, permission, or scientific validity.

| Endpoint | Exact file | Format / observed metadata | Size | SHA-256 | Git status |
| --- | --- | --- | ---: | --- | --- |
| PSI-107 fuel classification | `models/psi107/fuel_classifier_pipeline.joblib` | Joblib/pickle packet; reported contents include a PyTorch state dictionary, fitted `StandardScaler`, ordered input feature names, and class labels. | 13,614 bytes | `ad59bc8564dfbb1f9ba474ffcc4a80cf4b8cc56e84a9456e92be8959d4dadd27` | Ignored locally; not tracked |
| PSI-69 flame outcome | `models/psi69/dnn_cls_model.keras` | Keras archive; reported metadata declares Keras 3.13.2; eight inputs, three softmax outputs. Runtime compatibility currently uses TensorFlow 2.16.2 / Keras 3.10.0 with an in-memory compatibility shim. | 77,832 bytes | `8ef25ddfaab4f5899723c60c35df5dde75d11aa97d679f40e6971f02bf1fb32c` | Ignored locally; not tracked |
| PSI-69 flame outcome | `models/psi69/dnn_cls_preprocessor.joblib` | Joblib/pickle serialized preprocessing object; reported feature list includes pressure, initial O/N/CO/He fractions, droplet diameter, and fuel; output is eight transformed columns. | 4,239 bytes | `8f0ef7e8f76702f06f4b20455b0e4df57ec2dbff0de31a4e8a181d1ba392af57` | Ignored locally; not tracked |

The three files total 95,685 bytes. File size is not a GitHub constraint here; rights and provenance remain the blocker. Joblib files use pickle-based deserialization and must only be loaded from a trusted, integrity-verified source.

## Documentary evidence and unknowns

- Local integration notes identify an extracted `Predictive-Models-main (1).zip` archive with SHA-256 `04215418e96fd2af1fbc7769091c67fbbab85dcc1b4c4d5c7eb4b818fd89e729`. The archive itself, its source URL, and its license are not present in this checkout, so this is a recorded note rather than independently verified chain of custody.
- Repository documentation names `PSI-107_Experimental table_Spice.csv` and `PSI-69_Experimental table_FLEX.csv` as datasets in that extracted source. The source CSVs and original training code are not present here.
- PSI-107 local notes report a source-card accuracy of 0.8191 and a 469-row extracted table, but do not establish the evaluation split, exact train/test counts, dataset rights, or independently reproduced performance.
- PSI-69 local notes report a 274-row extracted table and an 80/20 split with SMOTE balancing, but no numeric evaluation metric. The model-card gas names conflict with the serialized preprocessor names (`O2/N2/CO2` versus `O/N/CO`), and class ordering is not supported by training code.
- A repository file search found no model-specific source license, `LICENSE`, `NOTICE`, or `COPYING` document for these artifacts. No public redistribution permission should be inferred from a model being downloadable or from its association with a NASA PSI experiment.
- The PSI-69 01-Model regression files are not part of this three-file inventory and remain excluded from the API; their target and units are undocumented.

## Clearance checklist — all items required before public distribution

- [ ] Record the original source repository or provider URL and exact release, commit, or archive identifier.
- [ ] Obtain the original license or written permission from the rights holder for each model, preprocessor, and associated metadata file.
- [ ] If public redistribution is not permitted, separately confirm that private storage, access, and production use are permitted under the applicable terms before placing the files in a private artifact store.
- [ ] Confirm the grant expressly permits redistribution of the serialized weights and preprocessing state in a public source repository and public container image; record any attribution, notice, share-alike, non-commercial, or other conditions.
- [ ] Confirm rights for training data and third-party material incorporated into model parameters or the preprocessor, separately from rights to the source code or dataset documentation.
- [ ] Confirm whether the artifact may be used in commercial services and by downstream users; document restrictions and required notices.
- [ ] Obtain and retain the original training/inference code, input schema, feature order, class-label order, preprocessing fit procedure, evaluation protocol, and model card from the rights holder.
- [ ] Reconcile PSI-69 feature-name and class-order discrepancies with authoritative training materials.
- [ ] Record an authorized maintainer, approval date, evidence location, license notice text, and the SHA-256 values approved for distribution.
- [ ] Review the pickle/joblib loading risk and accept the artifact supplier as trusted; keep SHA-256 verification enabled.
- [ ] Decide whether the public repository, public releases, and public container images may contain these exact hashes. Repeat clearance if the files change.

**Release gate:** until every applicable item is closed and evidence is retained, do not distribute the files. The current `.gitignore` and `.dockerignore` exclude all three exact paths, and the Dockerfile copies only an explicit allowlist that omits them. Its validation guard intentionally fails with the exact missing paths on a standard Git checkout. A private store does not establish permission by itself.

## Private artifact store distribution path

1. First establish that the rights holder permits the intended private use and delivery. Keep source code and this checklist in Git; only then store the exact approved model bytes under immutable/versioned object keys in an access-controlled artifact store.
2. Grant the build/deployment identity narrowly scoped, read-only access using short-lived OIDC/workload credentials or the platform's secret manager. Do not put long-lived credentials in Git, Docker build arguments, image layers, or logs.
3. The current `.dockerignore` and Dockerfile allowlist deliberately prevent these files from entering any build context/image, including a private build. After rights approval, make a separate reviewed change to permit only the approved paths, retrieve the approved object versions into a temporary private build context, verify every expected SHA-256, and fail the pipeline on any mismatch. Do not cache or publish the binary-bearing context as a public CI artifact.
4. Because this Dockerfile embeds the model files, publish the resulting image only to a private registry unless public redistribution rights have been cleared for all included artifacts. Restrict registry read access to the deployment service.
5. If a public image is required while model rights remain private, keep the models outside the image in a private mounted volume/store and make a separate reviewed application/Docker change to support that runtime path; the current build-time `COPY` and artifact check will not support an artifact-free public image.
6. Preserve the artifact version, source hash, approved destination, build identity, and resulting image digest in release records. Never log artifact contents or credentials.

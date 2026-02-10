## reduced-8  
reduced-8 uses the same training data as semantic-8 but only a small subset of test data, so that computationally demanding algorithms can also compete. Class labels are again {1: man-made terrain, 2: natural terrain, 3: high vegetation, 4: low vegetation, 5: buildings, 6: hard scape, 7: scanning artefacts, 8: cars}. An additional label {0: unlabeled points} marks points without ground truth and should not be used for training! Uniform downsampling with a resolution of 0.01 m was performed on the test set. Manipulations of the training set are left to the users.

reference: 
- [weblink](https://www.semantic3d.net/view_dbase.php?chl=2)
- [SEMANTIC3D.NET: A NEW LARGE-SCALE POINT CLOUD CLASSIFICATION BENCHMARK](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Papers/Hackel-etal-cmrt2017.pdf)


datalist:
- point clouds for testing as zipped ascii files:
    - MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt.7z( 0.12 GB )
    - StGallenCathedral_station6_rgb_intensity-reduced.txt.7z( 0.17 GB )
    - sg27_station10_rgb_intensity-reduced.txt.7z( 0.33 GB )
    - sg28_Station2_rgb_intensity-reduced.txt.7z( 0.29 GB )
- point clouds for training as zipped ascii files:
    - bildstein_station1_xyz_intensity_rgb.7z( 0.20 GB )
    - bildstein_station3_xyz_intensity_rgb.7z( 0.17 GB )
    - bildstein_station5_xyz_intensity_rgb.7z( 0.18 GB )
    - domfountain_station1_xyz_intensity_rgb.7z( 0.28 GB )
    - domfountain_station2_xyz_intensity_rgb.7z( 0.25 GB )
    - domfountain_station3_xyz_intensity_rgb.7z( 0.23 GB )
    - neugasse_station1_xyz_intensity_rgb.7z( 0.32 GB )
    - sg27_station1_intensity_rgb.7z( 1.87 GB )
    - sg27_station2_intensity_rgb.7z( 2.72 GB )
    - sg27_station4_intensity_rgb.7z( 1.59 GB )
    - sg27_station5_intensity_rgb.7z( 1.25 GB )
    - sg27_station9_intensity_rgb.7z( 1.22 GB )
    - sg28_station4_intensity_rgb.7z( 1.40 GB )
    - untermaederbrunnen_station1_xyz_intensity_rgb.7z( 0.17 GB )
    - untermaederbrunnen_station3_xyz_intensity_rgb.7z( 0.17 GB )
- ground truth labels for training as zipped ascii files ( 0.01 GB)



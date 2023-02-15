# Changelog

## [1.2.2](https://github.com/agrc/udot-parcel-ml/compare/v1.2.1...v1.2.2) (2023-02-15)


### 🐛 Bug Fixes

* handle code that throws ([d70bedd](https://github.com/agrc/udot-parcel-ml/commit/d70bedd347ee2816cc98edf338b0dc516c326c2e))


### 📖 Documentation Improvements

* add a missing step ([0233096](https://github.com/agrc/udot-parcel-ml/commit/0233096eadfc0a0adcdb580069fff4729e2f5052))
* add more information about the workflow ([157446d](https://github.com/agrc/udot-parcel-ml/commit/157446d25440286de061702195de3e67adcdb7d1))

## [1.2.1](https://github.com/agrc/udot-parcel-ml/compare/v1.2.0...v1.2.1) (2023-02-14)


### 📖 Documentation Improvements

* add the command for the workflow ([90ec950](https://github.com/agrc/udot-parcel-ml/commit/90ec950158f58a0b20568e5724a6c2c19dbab1dc))


### 🐛 Bug Fixes

* generate a new build ([dac5b78](https://github.com/agrc/udot-parcel-ml/commit/dac5b785f426bae81023a69fa38c432816a795a3))

## 1.2.0 (2023-02-14)


### 🎨 Design Improvements

* add doc string ([02c8fa2](https://github.com/agrc/udot-parcel-ml/commit/02c8fa29623aae605c98ab30c0965d639fc2272d))


### 🐛 Bug Fixes

* add another circle detection multipler ([cb2a0e6](https://github.com/agrc/udot-parcel-ml/commit/cb2a0e6447110f6414f4dd723ed9b3f50cecc69e))
* add missing object name to cli convert ([0221c8f](https://github.com/agrc/udot-parcel-ml/commit/0221c8f16616c64ee9d2ccc66bcd56826dd0f7ba))
* add periods to suffix list ([dc748e2](https://github.com/agrc/udot-parcel-ml/commit/dc748e2d1dd8deb9690df61c284ac33768f91220))
* bucket name has gs:// which needs to be removed ([7b63474](https://github.com/agrc/udot-parcel-ml/commit/7b63474eef680d6bb1bbabc9cc4deafa9fae2b4a))
* call correct method to upload bytes ([b957991](https://github.com/agrc/udot-parcel-ml/commit/b95799104a71680b11ca8f6af540a072006dadbf))
* catch bomb errors ([c27cfab](https://github.com/agrc/udot-parcel-ml/commit/c27cfabd418e7c4da1caaaef814c71fa80859aa8))
* check if the mosaic array has any value ([50c771e](https://github.com/agrc/udot-parcel-ml/commit/50c771e73580dc5115e6f935384596be4befc8df))
* convert pdf pages to bytes preserving image metadata ([d874a69](https://github.com/agrc/udot-parcel-ml/commit/d874a698be9dd83c1c3ddb35ea14da5cf1af2969))
* correct docopt syntax ([8fdee21](https://github.com/agrc/udot-parcel-ml/commit/8fdee213852ca455bf5d764bddfd82c1b5fa86e0))
* correct file saving after refactoring to bytes ([b0ff276](https://github.com/agrc/udot-parcel-ml/commit/b0ff276626d266320eca9fe758232e2650fe2ee0))
* correct method name ([6827300](https://github.com/agrc/udot-parcel-ml/commit/68273001b7e8296c4abc7e3d1e50e8749f15cc27))
* create a single item array of an images bytes ([cbb3dda](https://github.com/agrc/udot-parcel-ml/commit/cbb3dda9a57726f82590ba8ea961de5dd56343e8))
* create bounds on circle mask ([f7e656e](https://github.com/agrc/udot-parcel-ml/commit/f7e656e42547a472bcf34563d625c21ca9c27ca0))
* create parent folders for file index ([bcc7d29](https://github.com/agrc/udot-parcel-ml/commit/bcc7d2959b81f44bd5d3900949298d56fda33add))
* dispose of bytesio and use value ([da55494](https://github.com/agrc/udot-parcel-ml/commit/da554949661237dc5b70b9b69d6b68f7946ee192))
* fix logging issue where numbers were expected ([926032f](https://github.com/agrc/udot-parcel-ml/commit/926032f7b92f8b51e831d143822bb82d013d99e2))
* guard against empty images list ([3f45a42](https://github.com/agrc/udot-parcel-ml/commit/3f45a42ad75f4bee5ed791dca6726e31a617a8b8))
* guard against no circles found ([bf926e4](https://github.com/agrc/udot-parcel-ml/commit/bf926e49e86ea4347752348b25357056f1511362))
* insert buffered image ([4fc42c0](https://github.com/agrc/udot-parcel-ml/commit/4fc42c06603d9d7693332059c3b5fd62056d7563))
* make mosaic flag optional ([93bcdca](https://github.com/agrc/udot-parcel-ml/commit/93bcdca559af9a896f38b031df94338261c9bb07))
* pass string to constructor ([19b9f2d](https://github.com/agrc/udot-parcel-ml/commit/19b9f2dd121353247dde9944c5956629be6570b6))
* remove duplicate call to np.frombuffer ([18fbe21](https://github.com/agrc/udot-parcel-ml/commit/18fbe21f9fcebdbb7c6f7e98065e018a76c27c44))
* remove unnecessary path object ([79d3f22](https://github.com/agrc/udot-parcel-ml/commit/79d3f2274a04ce9584d15cf2d26d974dea8dfea8))
* return correct type ([92523e7](https://github.com/agrc/udot-parcel-ml/commit/92523e73c963225b138c4fd5484f05a5293cbd30))
* rstrip any newlines or spaces in file name ([e178744](https://github.com/agrc/udot-parcel-ml/commit/e1787445790feea999047607b953774443e93841))
* strip filenames ([34721c7](https://github.com/agrc/udot-parcel-ml/commit/34721c79ce2f1ff21f89002dda3a87a3ccf996fd))
* try to protect from bad pdf image decoding ([7cfe321](https://github.com/agrc/udot-parcel-ml/commit/7cfe321306de5b45b06e5bb80716fc000f7f89d4))


### 🚀 Features

* add ability to convert pdf's to pils ([cfb4e25](https://github.com/agrc/udot-parcel-ml/commit/cfb4e25bb07532743918e8a64d66b694bee58545))
* add function to get bucket file names ([7df81b8](https://github.com/agrc/udot-parcel-ml/commit/7df81b8a3435ce202a400a92f44cdcfe56741aab))
* add logging to row_run.py ([b22162b](https://github.com/agrc/udot-parcel-ml/commit/b22162bc14960b71e4e399a990b8576c6d304e01))
* add mosaic option ([0da4681](https://github.com/agrc/udot-parcel-ml/commit/0da46815a2ab9aebeeec454460409f329e4e7c4d))
* add perf_counter ([6877cbe](https://github.com/agrc/udot-parcel-ml/commit/6877cbe64a927182f39c42fd6a6662852e29761d))
* add upload_csv function ([b963218](https://github.com/agrc/udot-parcel-ml/commit/b9632185678eb5a9f70e65ecec65462b568f290e))
* add write_results function ([cda3563](https://github.com/agrc/udot-parcel-ml/commit/cda3563f1a669798c99115bf59b809f3dceb785a))
* build initial row_run.py ([c5712f8](https://github.com/agrc/udot-parcel-ml/commit/c5712f8aa8e7c384acab3ea3b2843e3a1cc28125))
* calculate remaining index ([1c7922a](https://github.com/agrc/udot-parcel-ml/commit/1c7922a00e6dc9f4353e6cbf20aebe88a7de002f))
* connect image convert cli command ([5174b8c](https://github.com/agrc/udot-parcel-ml/commit/5174b8c6af0aa6eb49c9434a7c7991ddc4cf1043))
* create cli method to create a file index ([cdfd64a](https://github.com/agrc/udot-parcel-ml/commit/cdfd64a73de9b489705d1181314f1b17f2a4ba03))
* create cli options to download, merge, and summarize results ([cf5055b](https://github.com/agrc/udot-parcel-ml/commit/cf5055b0cdce614eb6f2b725ab07b8598b4f382e))
* filter out deed documents ([b5f6f5a](https://github.com/agrc/udot-parcel-ml/commit/b5f6f5a6d9a9acc328de74990259c22b8a1b3f7b))
* implement ocr and circle detection methods ([2a3912b](https://github.com/agrc/udot-parcel-ml/commit/2a3912b4ee3867ac1f25666bac2e11993ebcbafd))


### 📖 Documentation Improvements

* add basic workflow steps ([12b3c45](https://github.com/agrc/udot-parcel-ml/commit/12b3c45e622704d4721b34403a76ead41d96c38f))
* add example for storage cli ([260d359](https://github.com/agrc/udot-parcel-ml/commit/260d35932cd7032383a3f578244f980bd1f50cee))
* add single image example ([4cae43a](https://github.com/agrc/udot-parcel-ml/commit/4cae43a089b85cb62511f0a657c73336959d027a))
* add some words about the CLI ([9f7568a](https://github.com/agrc/udot-parcel-ml/commit/9f7568a244c253c766d43bed261d714d51d4ef84))
* correct example file path ([ee032d6](https://github.com/agrc/udot-parcel-ml/commit/ee032d6558388681de794357dda5b5aab9ee356c))
* udpate usage for circle and ocr options ([0ab3dc4](https://github.com/agrc/udot-parcel-ml/commit/0ab3dc41a6e6c8276455d474ff674ee609544ccf))
* update function docstrings ([3edfa75](https://github.com/agrc/udot-parcel-ml/commit/3edfa75296520f3e9a8bcc38fde12de6b5241b6a))
* update readme ([88b2eb5](https://github.com/agrc/udot-parcel-ml/commit/88b2eb58f18d8501ed21aeec8ca9b95dde14685e))
* upload doc string style ([b419616](https://github.com/agrc/udot-parcel-ml/commit/b4196163c974ae4c2c2c545da97647cc04870e3f))

## [1.2.0-3](https://github.com/agrc/udot-parcel-ml/compare/v1.2.0-2...v1.2.0-3) (2023-02-07)


### 🐛 Bug Fixes

* remove unnecessary path object ([04dd9a8](https://github.com/agrc/udot-parcel-ml/commit/04dd9a811f6e3b3f63df6f6e3f107960d03c81b0))

## [1.2.0-2](https://github.com/agrc/udot-parcel-ml/compare/v1.2.0-1...v1.2.0-2) (2023-02-07)


### 🐛 Bug Fixes

* add missing object name to cli convert ([3105669](https://github.com/agrc/udot-parcel-ml/commit/310566969b873b6c1ad2d76b6c9c743c8b8b9f24))
* call correct method to upload bytes ([3adf908](https://github.com/agrc/udot-parcel-ml/commit/3adf908b1308d3aa6416bce2b1f3532238da4873))
* correct file saving after refactoring to bytes ([262874b](https://github.com/agrc/udot-parcel-ml/commit/262874be4a4c34b43b0a3dbde9411c86984bbb33))
* dispose of bytesio and use value ([8f85446](https://github.com/agrc/udot-parcel-ml/commit/8f85446d91dbf23148f0f7f18e710cd04901690a))
* guard against empty images list ([db7736f](https://github.com/agrc/udot-parcel-ml/commit/db7736f41dcccd04bd3de006af0af7c1982f996e))
* make mosaic flag optional ([428b4ac](https://github.com/agrc/udot-parcel-ml/commit/428b4acc6f7f7d1573f5884c66cabe57b2f6931a))
* pass string to constructor ([dcb45ff](https://github.com/agrc/udot-parcel-ml/commit/dcb45ff6b77a093789130a8ec6d27f926be5aaf3))
* return correct type ([2816048](https://github.com/agrc/udot-parcel-ml/commit/281604831f3d7f53808d184b6720cc996d1770ad))

## [1.2.0-1](https://github.com/agrc/udot-parcel-ml/compare/v1.2.0-0...v1.2.0-1) (2023-02-07)


### 🐛 Bug Fixes

* check if the mosaic array has any value ([7326ac4](https://github.com/agrc/udot-parcel-ml/commit/7326ac4a24d65a13495d7149e33e1985f977d085))

## [1.2.0-0](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-9...v1.2.0-0) (2023-02-06)


### 🚀 Features

* add mosaic option ([0cdcd63](https://github.com/agrc/udot-parcel-ml/commit/0cdcd636c78823605da8dbfd67dc7ed1290c645d))
* create cli options to download, merge, and summarize results ([dc90eee](https://github.com/agrc/udot-parcel-ml/commit/dc90eeeac8ae6dfcf3743ae771894e2afc482cba))


### 🐛 Bug Fixes

* catch bomb errors ([e4c67b4](https://github.com/agrc/udot-parcel-ml/commit/e4c67b4c6325c8a0b7f8f708b99e6582420a1f07))
* correct method name ([acf0c2b](https://github.com/agrc/udot-parcel-ml/commit/acf0c2b12a1d062c737a75ae107cc4999e63bd5b))
* insert buffered image ([bf12509](https://github.com/agrc/udot-parcel-ml/commit/bf12509767069d456faa7ec116b5be6b6d3f54c8))

## [1.0.0-9](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-8...v1.0.0-9) (2023-02-02)


### 🐛 Bug Fixes

* add another circle detection multipler ([bd4f04a](https://github.com/agrc/udot-parcel-ml/commit/bd4f04a055f747c4a331c618f684b4180e41e305))

## [1.0.0-8](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-7...v1.0.0-8) (2023-02-01)


### 🐛 Bug Fixes

* convert pdf pages to bytes preserving image metadata ([18c6589](https://github.com/agrc/udot-parcel-ml/commit/18c6589464ee5d9e6d83ac46d8b4801ff629f142))

## [1.0.0-7](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-6...v1.0.0-7) (2023-02-01)


### 🐛 Bug Fixes

* try to protect from bad pdf image decoding ([2217310](https://github.com/agrc/udot-parcel-ml/commit/2217310384c4ada44b910eef6f81c766e1db9e37))

## [1.0.0-6](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-5...v1.0.0-6) (2023-02-01)


### 🐛 Bug Fixes

* guard against no circles found ([a4bb1ce](https://github.com/agrc/udot-parcel-ml/commit/a4bb1ce1d950651cc248935104d8dc777e38c2ff))

## [1.0.0-5](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-4...v1.0.0-5) (2023-02-01)


### 🐛 Bug Fixes

* create a single item array of an images bytes ([ff1e06e](https://github.com/agrc/udot-parcel-ml/commit/ff1e06e96ddfc24d3dba436e11885350a9b0bc60))

## [1.0.0-4](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-3...v1.0.0-4) (2023-01-31)


### 🐛 Bug Fixes

* add periods to suffix list ([25d3a06](https://github.com/agrc/udot-parcel-ml/commit/25d3a06637ee46d140cc61713d00501e2bd3c61f))
* fix logging issue where numbers were expected ([c379a15](https://github.com/agrc/udot-parcel-ml/commit/c379a1595c6d41b75d8b9630dc140601ff5e8562))

## [1.0.0-3](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-2...v1.0.0-3) (2023-01-31)


### 🐛 Bug Fixes

* bucket name has gs:// which needs to be removed ([4130c24](https://github.com/agrc/udot-parcel-ml/commit/4130c24ea60209011ad6173f0fda1b95d9f64f30))
* rstrip any newlines or spaces in file name ([31f09ba](https://github.com/agrc/udot-parcel-ml/commit/31f09ba7b17bee88108d269beb00d4dbae9a191f))

## [1.0.0-2](https://github.com/agrc/udot-parcel-ml/compare/v1.0.0-1...v1.0.0-2) (2023-01-31)


### 🐛 Bug Fixes

* create parent folders for file index ([2ca84ac](https://github.com/agrc/udot-parcel-ml/commit/2ca84acae337ccd93640c5bd15822bb46049c7da))

## 1.0.0-0 (2023-01-31)


### 🐛 Bug Fixes

* correct docopt syntax ([ae62f2b](https://github.com/agrc/udot-parcel-ml/commit/ae62f2bcdf9af6d50b1a294a9adf786b06f24445))
* create bounds on circle mask ([58eb776](https://github.com/agrc/udot-parcel-ml/commit/58eb7766aa581a085e9f2d43594f7c8ad897950d))
* remove duplicate call to np.frombuffer ([5c5c8e5](https://github.com/agrc/udot-parcel-ml/commit/5c5c8e5ea64150090306fc39f762ba1a0366bcd9))


### 🚀 Features

* add ability to convert pdf's to pils ([9cf0888](https://github.com/agrc/udot-parcel-ml/commit/9cf0888db3956cae123eb8295bc2ce574821efa4))
* add function to get bucket file names ([a30fc0a](https://github.com/agrc/udot-parcel-ml/commit/a30fc0ae415d55b64f0609ee5be27149bd7a3d69))
* add logging to row_run.py ([fb6020c](https://github.com/agrc/udot-parcel-ml/commit/fb6020c883a7e797cc75ceb6777f50a36a4eafdd))
* add perf_counter ([c4bdc62](https://github.com/agrc/udot-parcel-ml/commit/c4bdc622b472b3323f477bd699ccf6c18d8e4158))
* add upload_csv function ([5b614e0](https://github.com/agrc/udot-parcel-ml/commit/5b614e0374b2fd7fb5fac45da98490fe3ddc281d))
* add write_results function ([d85c3d9](https://github.com/agrc/udot-parcel-ml/commit/d85c3d977477ae20ddb3db92bf1af465dae2763d))
* build initial row_run.py ([47d26a5](https://github.com/agrc/udot-parcel-ml/commit/47d26a54c6988d42f9b83b79d2082fa09abf2dc9))
* connect image convert cli command ([737860a](https://github.com/agrc/udot-parcel-ml/commit/737860ac8ce2ee4de3f08585cf4b7de1f2fd1450))
* create cli method to create a file index ([ccb7c75](https://github.com/agrc/udot-parcel-ml/commit/ccb7c755f0632aebc898e898e2404cfd86fc8997))
* implement ocr and circle detection methods ([7aa59a8](https://github.com/agrc/udot-parcel-ml/commit/7aa59a8daaec0438de760597160562a614e2a6ae))


### 🎨 Design Improvements

* add doc string ([b756fad](https://github.com/agrc/udot-parcel-ml/commit/b756fad0729173705f11cac1448313505ad35930))


### 📖 Documentation Improvements

* add example for storage cli ([8a81b1a](https://github.com/agrc/udot-parcel-ml/commit/8a81b1a35955733c89ae66845403548beaef62f2))
* add single image example ([555d163](https://github.com/agrc/udot-parcel-ml/commit/555d163a55209d86ce57774394cb834271c0d9f9))
* add some words about the CLI ([a4b0033](https://github.com/agrc/udot-parcel-ml/commit/a4b0033733819b2c049ee80ab3b26df06240a5e8))
* correct example file path ([3e228f2](https://github.com/agrc/udot-parcel-ml/commit/3e228f2cca8e55cfa254bf3ac38d0610fad68b9f))
* udpate usage for circle and ocr options ([b2446cc](https://github.com/agrc/udot-parcel-ml/commit/b2446cc1063ad85515133471ce849dec6dadacc3))
* update function docstrings ([2f6b417](https://github.com/agrc/udot-parcel-ml/commit/2f6b4179f745f7bcaced99f86a1390703b2d5fe7))
* upload doc string style ([f9877ac](https://github.com/agrc/udot-parcel-ml/commit/f9877ac4578346e033cbe734f065a8fe61da0824))

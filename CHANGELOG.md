# Changelog

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

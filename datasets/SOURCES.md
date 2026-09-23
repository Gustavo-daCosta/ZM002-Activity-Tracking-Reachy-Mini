# Dataset sources

HRNet COCO-17 2D skeleton annotations published by OpenMMLab for PySKL / MMAction2. No video downloads and
no pose estimation to run: each pickle holds `keypoint [M,T,V,2]`, `keypoint_score [M,T,V]`, `img_shape`,
`label`, `frame_dir` and the official splits. Re-download with `tools/download_datasets.sh`.

The pickles themselves are gitignored (1.8 GB).

## ntu60_hrnet.pkl — NTU RGB+D 60

- URL: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
- Size: 705 401 771 bytes — sha256 `953836d9dcae0b139ba524cee5003346c5cb4cbb26b0371cbf927f0cfa383c6a`
- Dataset: Shahroudy et al., "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", CVPR 2016.
  56 578 clips, 60 action classes, 40 subjects, 3 camera views, studio conditions.
- License: the ROSE Lab NTU RGB+D terms must be accepted to use the source dataset; these skeletons are
  redistributed by OpenMMLab. Academic / research use.
- Classes we use: label 22 = "A23 hand waving" (944 clips), label 9 = "A10 clapping". Every other class
  contributes negatives, capped per class.

## ucf101_hrnet.pkl — UCF101

- URL: https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl
- Size: 1 070 780 736 bytes — sha256 `5222399cb86d4e687db8437a20acaeee1bf1d8d4cc7a9036a31da0197529ed7e`
- Dataset: Soomro et al., "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild",
  CRCV-TR-12-01, 2012. 13 320 YouTube clips, 101 classes.
- License: research use only.
- Classes we use, resolved by clip name (`v_PushUps_g08_c02`), never by integer label: `PushUps`,
  `BodyWeightSquats`, `JumpingJack`. Every other class contributes negatives, capped per class.
- Caveat: in-the-wild footage — moving cameras, cuts, several people in frame — and the annotation tracks
  one person, not necessarily the one exercising. Expect worse per-class scores than on NTU.
- UCF101 verification: 13 320 clips total. All `frame_dir` values match the pattern `v_<Class>_g<NN>_c<NN>` (zero unmatched).
  Class counts for actions we use: `PushUps` 102, `BodyWeightSquats` 112, `JumpingJack` 123.

## hmdb51_2d.pkl — HMDB51

- URL: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
- Size: 212 759 219 bytes — sha256 `8d2c7a5348ed1c9897ea3c69aee025e2c17cc672c6f361683ce1784d0a784287`
- Dataset: Kuehne et al., "HMDB: A Large Video Database for Human Motion Recognition", ICCV 2011.
  6 371 clips, 51 classes, mostly movie footage.
- License: research use only.
- **External test set only — never trained on.** Classes used for evaluation: `wave`, `pushup`, and
  `situp`, which matters because it is the confusable neighbour of `pushup`.
- HMDB51 verification: 51 classes, 6 371 clips total. Alphabetical-index assumption confirmed against
  clip names: label 29 = `pushup` (103 clips), label 38 = `situp` (105), label 50 = `wave` (104), label 4 = `clap` (127).

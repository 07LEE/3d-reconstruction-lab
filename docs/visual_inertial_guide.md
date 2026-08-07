# Visual-Inertial (RGB + IMU) Pipeline Guide

본 문서는 **3DRC** 프로젝트에서 RGB 이미지 데이터와 함께 **IMU(Inertial Measurement Unit)** 센서 데이터를 결합하여 3D 재구성을 수행할 때의 기술적 장점 및 단계별 파이프라인 연동 가이드를 제공합니다.

---

## 1. Visual-Inertial Fusion의 주요 장점

단순 2D 이미지 기반 3D 재구성(Visual-Only SfM)과 비교하여 IMU 센서 데이터를 추가 활용할 때 얻을 수 있는 3가지 핵심 장점은 다음과 같습니다.

### (1) 실제 물리적 스케일(Metric Scale) 복원

* **Visual-Only 한계**: 단안(Monocular) 카메라 이미지만 사용 시 3D 공간의 스케일이 임의의 비율(Arbitrary Relative Scale)로 형성됩니다.
* **RGB + IMU 장점**: IMU의 가속도계가 측정하는 지구 중력 가속도($g \approx 9.81\,\mathrm{m/s^2}$) 기준을 통해 3D 포인트 클라우드 및 카메라 궤적의 실제 센치미터/미터(m) 단위 스케일을 정확히 복원합니다.

### (2) 중력 방향(Gravity Direction) 축 고정

* **Visual-Only 한계**: 3D 공간의 바닥면이나 수평/수직 축(Up Vector)이 첫 번째 카메라 프레임에 임의로 종속됩니다.
* **RGB + IMU 장점**: IMU가 상시 센싱하는 중력 벡터 방향을 3D 좌표계의 $Z$축(Down/Up Vector)으로 구속하여, `Planar-GS`의 바닥면 정렬 및 `SuGaR` 메시 추출 과정에서 좌표축 회전 오차를 최소화합니다.

### (3) 모션 블러 및 텍스처 결여 구간 강건성(Robustness) 향상

* **Visual-Only 한계**: 빠른 카메라 회전으로 인한 모션 블러(Motion Blur) 또는 무늬가 없는 벽면(Textureless Region)을 지날 때 특징점 매칭 실패로 궤적 추적이 상실(Tracking Lost)됩니다.
* **RGB + IMU 장점**: IMU는 조명이나 텍스처 변화에 영향을 받지 않고 고주파수(100~1000Hz)로 관성 데이터를 제공하므로, 시각 특징점 추적이 불안정한 구간에서도 궤적 추정을 안정적으로 유지합니다.

---

## 2. 3DRC 프로젝트 내 파이프라인 적용 방식

```text
[RGB Image + IMU Data]
       ↓ (Step 1: vi_sfm)
[Gravity & Metric-Scale Aligned COLMAP Model]
       ↓ (Step 2: train_3dgs)
[Metric-Scale 3D Gaussian Asset]
       ↓ (Step 4: train_sugar)
[Gravity-Aligned Polygon Mesh (.obj)]
```

### Step 1: Visual-Inertial Camera Pose Estimation (`vi_sfm`)

* `scripts/01_sfm_hloc.sh vi_sfm` 명령어를 통해 구동합니다.
* SuperPoint + SuperGlue 특징점 추적 결과와 IMU 사전 적분 데이터를 합성하여 정밀한 카메라 포즈 및 희소 포인트 클라우드(`data/vi_sfm_reconstruction/sparse/0`)를 생성합니다.

### Step 2: Metric-Scale 3DGS Training

* `vi_sfm`으로 생성된 3D 포즈 데이터셋을 기반으로 `scripts/02_train_3dgs.sh`를 구동합니다.
* 학습된 3D Gaussian 모델은 실제 비율(Metric Scale)을 반영하므로 로보틱스, AR/VR 환경에 즉시 배치할 수 있습니다.

### Step 4: SuGaR Mesh Reconstruction

* `scripts/04_train_sugar.sh`를 실행하여 3DGS 모델로부터 메시(Polygon Mesh)를 추출합니다.
* 중력 방향 축이 사전에 완벽히 보정되어 있으므로 메시에 수평 바닥면 정렬이 자동으로 반영됩니다.

---

## 3. IMU 데이터 포맷 및 준비 가이드

프로젝트 내 `vi_sfm` 파이프라인은 아래와 같은 CSV 또는 EuroC 포맷의 IMU 데이터를 지원합니다.

### 지원 데이터 포맷 (EuroC CSV Format)

파일 경로: `data/imu_data.csv` (또는 `configs/default_config.sh` 내 `IMU_DATA_PATH` 지정)

```csv
#timestamp_ns,w_x,w_y,w_z,a_x,a_y,a_z
1403636580000000000,0.012,0.002,-0.005,0.120,0.050,9.810
1403636580005000000,0.015,0.001,-0.004,0.118,0.052,9.808
...
```

* `timestamp_ns`: 나노초(ns) 또는 초(s) 단위 타임스탬프
* `w_x, w_y, w_z`: 각속도(Angular Velocity, rad/s)
* `a_x, a_y, a_z`: 선가속도(Linear Acceleration, m/s^2)

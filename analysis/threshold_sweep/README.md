# How to run ablation study and analysis

First run an ablation run e.g. 
for stage 1:

```bash
bash utils/run_stage1_ablation.sh
```

for stage 2:
```bash
bash utils/run_stage2_ablation.sh
```

## Create caches and inference runs: 

e.g.
```bash
MODEL_DIR_STAGE1=... path_to_stage1_ablation_run.../runs/ablations_stage1_A0_A3_20260111_073857 \
MODEL_NAME_STAGE1=A2_plus_aug_p0.8_plus_focal_gamma2.0 \
MODEL_DIR_STAGE2=... path_to_stage2_ablation_run.../runs/ablations_stage2_A0_A3_20260111_015844 \
MODEL_NAME_STAGE2=A2_plus_aug_p0.8_plus_focal_gamma2.0 \
LONG_AUDIO_ROOT=... path_to_long_audio_root.../New_SwallowSet/Long \
EXPERIMENT_TAG=ablation_study PIPELINE_TAG=s1A2_s2A2 \
bash utils/run_export_window_probs_testset_mixed.sh
```


for single stage multiclass:
```bash
MODEL_DIR=... path_to_multiclass_run.../runs/ast_classifier_multiclass_valsplit \
LONG_AUDIO_ROOT=... path_to_long_audio_root.../New_SwallowSet/Long \
EXPERIMENT_TAG=ast_multiclass PIPELINE_TAG=ast_multiclass_valsplit bash src/run_export_window_probs_testset_multiclass.sh
```

## Run threshold analysis

e.g. for a given pipeline with given thresholds:
```bash 
python analysis/threshold_sweep/summarize_patient_level_zsr.py   --long-audio-root ... path_to_long_audio_root.../New_SwallowSet/Long   --cache-root ... path_to_cache_root.../caches/ablation_study   --pipelines s1A0_s2A0   --t1 0.5 --t2 0.5 --tzsr 0.5
```

## plot metrics vs tzsr based on threshold analysis:

```bash
python analysis/threshold_sweep/plot_zsr_sweep_metrics.py \
  --results-root analysis/outputs/patient_level_zsr \
  --pipelines s1A1b_s2A1b \
  --t1-values 0.5 --t2-values 0.5 \
```

further customization:

```bash
python analysis/threshold_sweep/plot_zsr_sweep_metrics.py \
  --results-root analysis/outputs/patient_level_zsr \
  --pipelines s1A1b_s2A1b \
  --t1-values 0.5 --t2-values 0.5 \
  --aggregations mean \
  --metrics recall,specificity,accuracy \
  --xtick-step 0.1 \
  --legend-inside --legend-inside-loc lower \
  --label-fontsize 14 \
  --title "Patient-level metrics vs ZSR threshold" \
  --title-fontsize 16 \
  --out analysis/outputs/patient_level_zsr/zsr_sweep_metrics.png
```



## Visual inspection of single waveplot via:

```bash
python analysis/threshold_sweep/plot_wave_detections.py   --patient-dir ... path_to_cache_root.../caches/<experiment>/<pipeline>/fold1/001/  --t1 0.5 --t2 0.5 --tzsr 0.5   --out-dir ... path_to_out_dir.../analysis/outputs/wave_plots/<pipeline>/
```


### Multi-class analysis

```bash
python analysis/threshold_sweep/plot_wave_detections_multiclass.py \
  --patient-dir caches/<experiment>/<pipeline>/fold2/010 \
  --mode zsr --t1 0.5 --t2 0.5 --tzsr 0.3 \
  --out-dir analysis/outputs/multiclass_wave_plots
```

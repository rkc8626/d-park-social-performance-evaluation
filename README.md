# Park Activity Video Analytics Dataset Plan

## Goal

Build a video-based dataset and analytics pipeline to measure park usage, movement, activities, social interaction, and nature exposure from fixed outdoor camera videos.

The final dataset should support hourly summaries of:

- Bike activity
- Pedestrian activity
- Visitor counts
- Time spent in the park
- Distance to trees or lake as a proxy for nature exposure
- Common activity types
- Group presence and social interaction proxy
- Apparent child percentage as a proxy for age-group friendliness

## 1. Target Outputs

| Metric | Output | Method | Reliability |
|---|---|---|---|
| Hourly bike activity | Number of bikers per hour | Detect and track bicycles/cyclists | High |
| Hourly pedestrian activity | Number of pedestrians per hour | Detect and track persons | High |
| Visitor counts | Unique tracked visitors per hour | Track ID counting inside park ROI | High |
| Time spent in park | Dwell time per visitor | Track duration inside park ROI | Medium-High |
| Nature exposure proxy | Distance/time near tree or lake | Person-to-ROI distance | Medium |
| Activity range | Walking, biking, sitting, talking, etc. | Rule-based labels plus activity classifier | Medium |
| Social connection proxy | Group size and group duration | Distance plus movement similarity | Medium |
| Child percentage | Apparent child / total visible people | Manual labels plus classifier | Medium-Low |

Important: “Feeling connected to nature” and “getting to know new people” cannot be directly measured from video. They should be reported as visual proxies, not direct psychological outcomes.

## 2. Dataset Structure

```text
park_video_dataset/
|
├── raw_videos/
│   ├── camera01_2026-05-23_0900-1000.mp4
│   ├── camera01_2026-05-23_1000-1100.mp4
│   └── ...
|
├── frames/
│   ├── camera01/
│   └── camera02/
|
├── annotations/
│   ├── boxes/
│   ├── tracks/
│   ├── activities/
│   ├── age_group_labels/
│   └── roi_polygons/
|
├── outputs/
│   ├── tracks.csv
│   ├── events.csv
│   ├── hourly_metrics.csv
│   └── visualizations/
|
└── README.md
```

## 3. Annotation Plan

| Annotation Type | Labels | Tool | Purpose |
|---|---|---|---|
| Object detection | person, bicycle, cyclist, stroller, dog | CVAT / Label Studio | Count people and bikes |
| Tracking | track_id for each person/biker | CVAT / auto-tracking | Visitor count and dwell time |
| Park zones | path, grass, lake, tree area, pavilion, entrance, exit | Manual polygon | Zone-based analysis |
| Activity labels | walking, running, biking, sitting, standing, talking, playing, dog walking, exercising, picnic | CVAT / Label Studio | Activity diversity |
| Age proxy | apparent_child, adult, uncertain | Manual label | Child percentage |
| Group label | group_id | Manual plus rule-based | Social interaction proxy |

## 4. Video Sampling Strategy

| Sampling Unit | Plan |
|---|---|
| Cameras | Use all available fixed camera views |
| Days | Sample weekdays and weekends |
| Time periods | Morning, noon, afternoon, evening |
| Weather | Include sunny, cloudy, and low-light conditions if available |
| Annotation density | Start with 5–10 minutes per hour |
| Initial labeled data | 500–2,000 frames for detection; 200–500 short clips for activity labels |

## 5. Model Pipeline

| Step | Model / Method | Output |
|---|---|---|
| Frame extraction | FFmpeg / OpenCV | Timestamped frames |
| Object detection | YOLO / RT-DETR | Person, bicycle, cyclist boxes |
| Multi-object tracking | ByteTrack / BoT-SORT | Consistent track IDs |
| ROI segmentation | Manual polygons; optional SAM/SAM2 | Lake, tree, path, grass regions |
| Activity recognition | Rule-based first; later MMAction2 / VideoMAE / SlowFast | Activity label per track segment |
| Group detection | Distance plus motion similarity | Group ID and group size |
| Child proxy | Manual labels first; later fine-tuned classifier | apparent_child / adult / uncertain |
| Metric aggregation | Python / Pandas | Hourly metrics table |

## 6. Core Data Tables

### `videos.csv`

| Column | Description |
|---|---|
| video_id | Unique video ID |
| camera_id | Camera/view ID |
| start_time | Real-world video start time |
| end_time | Real-world video end time |
| fps | Frame rate |
| resolution | Video resolution |
| location | Park/camera location |
| notes | Weather, lighting, camera issues |

### `tracks.csv`

| Column | Description |
|---|---|
| video_id | Source video |
| timestamp | Frame timestamp |
| track_id | Unique object track |
| class | person / bicycle / cyclist |
| bbox_x1, bbox_y1, bbox_x2, bbox_y2 | Bounding box |
| zone | Current park zone |
| speed | Estimated movement speed |
| apparent_age_group | child / adult / uncertain |
| group_id | Group ID if detected |
| confidence | Model confidence |

### `events.csv`

| Column | Description |
|---|---|
| video_id | Source video |
| track_id | Visitor track ID |
| start_time | Activity start |
| end_time | Activity end |
| activity_label | Activity type |
| zone | Activity zone |
| group_id | Group ID if applicable |

### `hourly_metrics.csv`

| Column | Description |
|---|---|
| hour | Aggregation hour |
| pedestrian_count | Unique pedestrian tracks |
| biker_count | Unique biker/cyclist tracks |
| visitor_count | Unique person tracks inside park ROI |
| median_dwell_time | Median time spent in park |
| total_visitor_minutes | Sum of all visitor dwell time |
| pct_near_tree | Percentage of visitors near tree ROI |
| pct_near_lake | Percentage of visitors near lake ROI |
| avg_distance_to_tree | Average person-to-tree distance |
| avg_distance_to_lake | Average person-to-lake distance |
| activity_diversity | Number/distribution of activity types |
| group_visitor_pct | Percentage of visitors in groups |
| mean_group_size | Average group size |
| apparent_child_pct | Percentage of visible people labeled apparent_child |

## 7. Processing Plan

| Phase | Task | Output |
|---|---|---|
| Phase 1 | Organize videos with camera ID and timestamps | Clean video inventory |
| Phase 2 | Extract frames and sample clips | Frame dataset |
| Phase 3 | Draw ROI polygons for each camera view | Zone definitions |
| Phase 4 | Manually label initial frames and clips | Ground-truth annotations |
| Phase 5 | Run YOLO detection plus ByteTrack/BoT-SORT tracking | Object tracks |
| Phase 6 | Compute hourly pedestrian, biker, and visitor counts | Count metrics |
| Phase 7 | Compute dwell time and zone occupancy | Park usage metrics |
| Phase 8 | Compute distance to lake/tree ROI | Nature exposure proxy metrics |
| Phase 9 | Detect groups using spatial-temporal rules | Social interaction proxy |
| Phase 10 | Label and classify common activities | Activity metrics |
| Phase 11 | Validate model outputs against manual labels | Error report |
| Phase 12 | Export final CSV files and dashboard plots | Final dataset |

## 8. Activity Label Set

| Label | Definition |
|---|---|
| walking | Person moving slowly on path or grass |
| running | Person moving faster than walking |
| biking | Person riding a bicycle |
| sitting | Person stationary in seated posture |
| standing | Person stationary but upright |
| talking/socializing | Two or more people close together and stationary or moving together |
| playing | Child or adult engaged in recreational movement |
| dog_walking | Person walking with visible dog |
| exercising | Stretching, workout, sports-like movement |
| picnic/resting | Stationary group or person on grass/pavilion area |

## 9. Evaluation Plan

| Component | Metric |
|---|---|
| Person detection | Precision, recall, mAP |
| Bicycle/cyclist detection | Precision, recall, mAP |
| Tracking | ID switches, track fragmentation, MOTA/IDF1 |
| Visitor count | Error vs. manually counted visitors |
| Dwell time | Mean absolute error vs. manual track duration |
| Activity labels | Accuracy/F1 on labeled clips |
| Child proxy | Accuracy/F1 with child/adult/uncertain labels |
| Group detection | Precision/recall against manually labeled groups |

## 10. Recommended Baseline Implementation

| Component | Recommended Choice |
|---|---|
| Annotation tool | CVAT |
| Detection model | YOLOv8/YOLOv11 or RT-DETR |
| Tracking model | ByteTrack or BoT-SORT |
| ROI annotation | Manual polygon per camera |
| Activity recognition | Rule-based baseline first |
| Advanced activity model | MMAction2 / VideoMAE / SlowFast |
| Segmentation helper | SAM/SAM2 if ROI annotation needs automation |
| Data processing | Python, OpenCV, Pandas |
| Visualization | Plotly / Matplotlib / Streamlit |


## 12. Key Notes

- Use “apparent child” instead of “child” to avoid claiming true age.
- Use “nature exposure proxy” instead of “feeling connected to nature.”
- Use “group/social interaction proxy” instead of “getting to know new people.”
- Do not report raw identities or face-level information.
- Blur faces if releasing public videos or frames.
- Always report model error rates with the final metrics.

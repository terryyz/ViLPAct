#ViLPAct Code Base Description


## Crowdsourcing
* [Huamn Evaluation of All Action Sequences](crowdsourcing/action_sequence_evaluation_planning.html): A front-end file for displaying the crowdsourcing process.
* [Activity Intent Annotation](crowdsourcing/intention_task.html):

## Data
* [Dataset](data/dataset-split): There are 1921 videos in `train.csv`, 481 videos in `val.csv`, 511 videos in `test.csv`.
* [Ground-truth Action Sequences](data/gt_act_seq): Generally, all `*_gt_seq.csv` contains the columns of 
  * `id` (video IDs), 
  * `action_flow_id` (in the format of `action_id starting_time ending_time;action_id starting_time ending_time`), 
  * `action_flow_des` (action flow in text based on `action_flow_id`), `intention_1` (in the format of `low_level_intent;high_level_intent`),
  * `intention_2` (in the format of `low_level_intent;high_level_intent`), 
  * `act_seq_start` (action flow starting time),
  * `act_seq_end` (action flow ending time),
  * `ob_start_frame` (starting index of frames in the observation),
  * `ob_end_frame` (ending index of frames in the observation),
  * `ob_act_seq` (observed action sequence part),
  * `future_act_seq` (unobserved action sequence part to be planned),
  * `scene` (the place of the human activity)
* [MKB](data/KB):
  * `action_class.csv`: symbolic knowledge of all actions in multimodal knowledge base.
  * `action_video_clip.csv`: information of all actions inside all videos
  * `video_activity.csv`: the same information as the one in `*_gt_seq.csv`
  * `features/`: all features in multimodal knowledge base
* [Prototype Feature](data/prototype_feature)
* [MQA](data/qa_evaluation)

## Model
* [Deep Generative Baselines](model/deep_generative_baselines): scripts for generating top-k action sequences and corresponding log probabilities.
* [NSPlan](model/NSPlan): implementation of the neurosymbolic planning model
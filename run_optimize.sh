prediction_dir="C:/Users/bellmi2/Documents/Praxissemester/struct2depth/output"
model_ckpt="C:/Users/bellmi2/Documents/Praxissemester/struct2depth/model/model-199160"
handle_motion="true"
size_constraint_weight="1" # This must be zero when not handling motion.

# If running on KITTI, set as follows:
data_dir="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02"
triplet_list_file="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/train.txt"
ft_name="kitti"



C:/Users/bellmi2/Anaconda3/envs/tf/python.exe optimize.py \
  --logtostderr \
  --output_dir $prediction_dir \
  --data_dir $data_dir \
  --triplet_list_file $triplet_list_file \
  --ft_name $ft_name \
  --model_ckpt $model_ckpt \
  --file_extension png \
  --handle_motion $handle_motion \
  --size_constraint_weight $size_constraint_weight
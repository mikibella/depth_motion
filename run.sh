input_dir="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/train.txt"
output_dir="C:/Users/bellmi2/Documents/Praxissemester/struct2depth/output"
model_checkpoint="C:/Users/bellmi2/Documents/Praxissemester/struct2depth/model/model-199160"

C:/Users/bellmi2/Anaconda3/envs/tf/python.exe inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion true \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
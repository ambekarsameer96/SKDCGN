
function train {
	# file path
	bs=$1
	save_dir='gan'
	IS_dir='/hdd2/tera/stat_real'
	FID_dir='Get_ImageNet_FID'
	img_dir='out_img'
	z_path='all_noises.npy'
	lb_path='all_labels.npy'
	image_size=256

	cmd=(python3 main.py --real_dir $2
		--batch_siz ${bs}
		--save_dir ${save_dir}
		--real_incep_stat_dir ${IS_dir}
		--real_fid_stat_dir ${FID_dir}
		--image_dir ${img_dir}
		--z_path ${z_path}
		--label_path ${lb_path}
		--mode train
		--image_size ${image_size})


	CUDA_VISIBLE_DEVICES=0 ${cmd[@]}
}

train 16 $1

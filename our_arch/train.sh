
function train {
	# file path
	bs=$1
	save_dir='gan'
	IS_dir='/hdd2/tera/stat_real'
	FID_dir='Get_ImageNet_FID'
	real_dir='/hdd2/tera/train_128'
#	img_dir='/hdd2/tera/ImageNet_1000/out_img+/hdd2/tera/ImageNet_1000_2/out_img' #saved imgs
#	z_path='/hdd2/tera/ImageNet_1000/all_noises.npy+/hdd2/tera/ImageNet_1000_2/all_noises.npy'
#	lb_path='/hdd2/tera/ImageNet_1000/all_labels.npy+/hdd2/tera/ImageNet_1000_2/all_labels.npy'
	img_dir='generated_images' #saved imgs
	z_path='all_noises.npy'
	lb_path='all_labels.npy'
	image_size=256
	
	cmd=(python3 main.py 
		--batch_siz ${bs}
		--save_dir ${save_dir} 
		--real_incep_stat_dir ${IS_dir} 
		--real_fid_stat_dir ${FID_dir}
		--real_dir ${real_dir}
		--image_dir ${img_dir} 
		--z_path ${z_path} 
		--label_path ${lb_path}
		--image_size ${image_size}
		--mode train)
		
	CUDA_VISIBLE_DEVICES=0 ${cmd[@]}
}



train 32

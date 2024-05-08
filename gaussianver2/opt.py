import configargparse

def create_parser(cmd=None, mode='config'):
    parser = configargparse.ArgumentParser(description='Splatfacto Model Configurations')
    
    # 各フィールドをコマンドライン引数として追加
    parser.add_argument('--camera_data_path', type=str)
    parser.add_argument('--image_data_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--sparse_model_path', type=str)
    parser.add_argument('--resize', default=False,action='store_true')
    parser.add_argument('--only_mesh',default=False,action='store_true')
    parser.add_argument('--max_steps', type=int, default=4000)
    parser.add_argument('--warmup_length', type=int, default=500, help='period of steps where refinement is turned off')
    parser.add_argument('--refine_every', type=int, default=100, help='period of steps where gaussians are culled and densified')
    parser.add_argument('--resolution_schedule', type=int, default=250, help='training starts at 1/d resolution, every n steps this is doubled')
    parser.add_argument('--background_color', type=str, choices=['random', 'black', 'white'], default='white', help='Whether to randomize the background color.')
    parser.add_argument('--num_downscales', type=int, default=0, help='at the beginning, resolution is 1/2^d, where d is this number')
    parser.add_argument('--cull_alpha_thresh', type=float, default=0.005, help='threshold of opacity for culling gaussians')
    parser.add_argument('--cull_scale_thresh', type=float, default=0.5, help='threshold of scale for culling huge gaussians')
    parser.add_argument('--continue_cull_post_densification', action='store_true', help='If True, continue to cull gaussians post refinement')
    parser.add_argument('--reset_alpha_every', type=int, default=30, help='Every this many refinement steps, reset the alpha')
    parser.add_argument('--densify_grad_thresh', type=float, default=0.0002, help='threshold of positional gradient norm for densifying gaussians')
    parser.add_argument('--densify_size_thresh', type=float, default=0.01, help='below this size, gaussians are duplicated, otherwise split')
    parser.add_argument('--n_split_samples', type=int, default=4, help='number of samples to split gaussians into')
    parser.add_argument('--sh_degree_interval', type=int, default=1000, help='every n intervals turn on another sh degree')
    parser.add_argument('--cull_screen_size', type=float, default=0.15, help='if a gaussian is more than this percent of screen space, cull it')
    parser.add_argument('--split_screen_size', type=float, default=0.05, help='if a gaussian is more than this percent of screen space, split it')
    parser.add_argument('--stop_screen_size_at', type=int, default=10000, help='stop culling/splitting at this step WRT screen size of gaussians')
    parser.add_argument('--random_init', action='store_true', help='whether to initialize the positions uniformly randomly')
    parser.add_argument('--num_random', type=int, default=50000, help='Number of gaussians to initialize if random init is used')
    parser.add_argument('--random_scale', type=float, default=10.0, help='Size of the cube to initialize random gaussians within')
    parser.add_argument('--ssim_lambda', type=float, default=0.2, help='weight of ssim loss')
    parser.add_argument('--stop_split_at', type=int, default=3500, help='stop splitting at this step')
    parser.add_argument('--sh_degree', type=int, default=2, help='maximum degree of spherical harmonics to use')
    parser.add_argument('--use_scale_regularization', action='store_true', help='If enabled, use scale regularization for reducing huge spikey gaussians.')
    parser.add_argument('--max_gauss_ratio', type=float, default=10.0, help='threshold of ratio of gaussian max to min scale before applying regularization loss')

    return parser
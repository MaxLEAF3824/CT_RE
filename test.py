from thop import profile
from network import *
from data_loader import *
from tqdm import tqdm
from evaluation import *
from torchvision.utils import save_image

# 注意：1.填写的数据集路径下直接包含图片，不要有子文件夹 2.填写Quarter_Dose文件夹的路径
test_path='data/TestSet/Quarter_Dose'
model_path = 'model/AttU_Net-100-0.0004-70-0.0.pkl'


# 加载数据集
test_loader = get_loader(image_path=test_path,image_size=256,batch_size=1,num_workers=1,mode='test',augmentation_prob=0.,shuffle=False)

# 加载模型
model = AttU_Net(img_ch=1,output_ch=1)
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 计算GFLOPS和参数量
input_sample = torch.randn(1, 1, 256, 256).to(device)
flops, params = profile(model, inputs=(input_sample,))
gflops = flops / 1e9
num_params = sum([param.numel() for param in model.parameters()])
print(f"#Parameters: {num_params} GFLOPS: {gflops:.2f}".format(gflops))

# 重构测试集图像
model.eval()
torch.set_grad_enabled(False)
save_dir = './test_qd_rc'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
psnr = 0
ssim = 0
for i, (images, GT) in tqdm(enumerate(test_loader),total=len(test_loader)):
    images = images.to(device)
    SR = model(images)
    psnr += get_psnr(SR,GT) / len(test_loader)
    ssim += get_ssim(SR,GT) / len(test_loader)
    save_image(SR.data.cpu(),f'{save_dir}/{i}.png')

print(f'PSNR: {psnr:.2f} SSIM: {ssim:.4f}')
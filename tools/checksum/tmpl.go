package main

var files = [...]string{
	"hubert/hubert_base.pt",
	"rmvpe/rmvpe.pt",
	"rmvpe/rmvpe.onnx",

	"pretrained/D32k.pth",
	"pretrained/D40k.pth",
	"pretrained/D48k.pth",
	"pretrained/G32k.pth",
	"pretrained/G40k.pth",
	"pretrained/G48k.pth",
	"pretrained/f0D32k.pth",
	"pretrained/f0D40k.pth",
	"pretrained/f0D48k.pth",
	"pretrained/f0G32k.pth",
	"pretrained/f0G40k.pth",
	"pretrained/f0G48k.pth",

	"pretrained_v2/D32k.pth",
	"pretrained_v2/D40k.pth",
	"pretrained_v2/D48k.pth",
	"pretrained_v2/G32k.pth",
	"pretrained_v2/G40k.pth",
	"pretrained_v2/G48k.pth",
	"pretrained_v2/f0D32k.pth",
	"pretrained_v2/f0D40k.pth",
	"pretrained_v2/f0D48k.pth",
	"pretrained_v2/f0G32k.pth",
	"pretrained_v2/f0G40k.pth",
	"pretrained_v2/f0G48k.pth",

}

const envtmpl = `sha256_hubert_base_pt = %s
sha256_rmvpe_pt      = %s
sha256_rmvpe_onnx    = %s

sha256_v1_D32k_pth   = %s
sha256_v1_D40k_pth   = %s
sha256_v1_D48k_pth   = %s
sha256_v1_G32k_pth   = %s
sha256_v1_G40k_pth   = %s
sha256_v1_G48k_pth   = %s
sha256_v1_f0D32k_pth = %s
sha256_v1_f0D40k_pth = %s
sha256_v1_f0D48k_pth = %s
sha256_v1_f0G32k_pth = %s
sha256_v1_f0G40k_pth = %s
sha256_v1_f0G48k_pth = %s

sha256_v2_D32k_pth   = %s
sha256_v2_D40k_pth   = %s
sha256_v2_D48k_pth   = %s
sha256_v2_G32k_pth   = %s
sha256_v2_G40k_pth   = %s
sha256_v2_G48k_pth   = %s
sha256_v2_f0D32k_pth = %s
sha256_v2_f0D40k_pth = %s
sha256_v2_f0D48k_pth = %s
sha256_v2_f0G32k_pth = %s
sha256_v2_f0G40k_pth = %s
sha256_v2_f0G48k_pth = %s

`

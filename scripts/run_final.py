# run_final.py
import torch
from diffusers import ZImagePipeline

# 공식 README에 명시된 ID
repo_id = "Tongyi-MAI/Z-Image-Turbo"

print(f"🚀 [{repo_id}] 모델 로딩 중 (Diffusers 방식)...")

try:
    # 1. 파이프라인 로드
    # trust_remote_code=True: 이 모델은 최신이라 허깅페이스 Hub의 코드를 받아와야 실행됩니다.
    pipe = ZImagePipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,  # 3090 VRAM 절약
        low_cpu_mem_usage=False,
        # trust_remote_code=True,      # 필수 옵션 (README 참조)
        # use_safetensors=True
    )
    
    # 2. GPU 할당
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)

    # 3. 이미지 생성
    # prompt = "A cinematic shot of a giant cat fighting a Gundam robot in Tokyo city, rain, neon lights, 8k, hyper-realistic"
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )
    prompt = ""
    print("🎨 이미지 생성 시작...")
    
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9, # 퀄리티를 위해 스텝 수 조정 가능
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42) # 재현성 있는 결과
    ).images[0]

    # 4. 저장
    image.save("final_result.png")
    print("✅ 성공! final_result.png 파일을 확인하세요.")

except AttributeError as e:
    print(f"\n❌ [치명적 오류] PyTorch가 손상되었습니다: {e}")
    print("👉 해결책: Visual C++ Redistributable 설치 후 반드시 '재부팅'을 하셔야 합니다.")
except Exception as e:
    print(f"\n❌ 실행 중 오류 발생: {e}")
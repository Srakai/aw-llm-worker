"""Inference backends for vision language models."""

import os
import logging
import subprocess
import shutil
from typing import Any, Dict, Optional
from awllm.prompt import SYSTEM_PROMPT, COARSE_ENUM, build_user_prompt, extract_json

LOG = logging.getLogger("aw-llm-worker")

# Try to import llama_cpp
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    LOG.warning("llama-cpp-python not available, only CLI mode will work")


class QwenVLPython:
    """Qwen2.5-VL inference using llama-cpp-python library."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temp: float = 0.2,
        max_tokens: int = 256,
        threads: int = 0,
        verbose: bool = False,
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available")

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.temp = temp
        self.max_tokens = max_tokens

        # Initialize chat handler with the mmproj (CLIP) model
        chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=verbose)

        # Initialize Llama with the chat handler
        self.llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=threads,
            logits_all=False,
            verbose=verbose,
        )

    def classify(
        self, img_path: str, meta: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify screenshot and return structured label."""
        sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": build_user_prompt(meta, ctx)}
        
        out = self.llm.create_chat_completion(
            messages=[sys_msg, user_msg],
            temperature=self.temp,
            max_tokens=self.max_tokens,
        )
        
        txt = out["choices"][0]["message"]["content"]
        obj = extract_json(txt)
        return self._normalize(obj)

    def _normalize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate LLM output."""
        obj["coarse_activity"] = str(obj.get("coarse_activity", "misc")).lower()
        if obj["coarse_activity"] not in COARSE_ENUM:
            obj["coarse_activity"] = "misc"
        
        # Clamp confidences
        try:
            obj["confidence"] = float(max(0.0, min(1.0, obj.get("confidence", 0.0))))
        except Exception:
            obj["confidence"] = 0.0
        
        try:
            pj = obj.get("project") or {}
            pj["confidence"] = float(max(0.0, min(1.0, pj.get("confidence", 0.0))))
            obj["project"] = pj
        except Exception:
            obj["project"] = {"name": None, "confidence": 0.0, "reason": ""}
        
        return obj


class QwenVLCLI:
    """Qwen2.5-VL inference using llama-mtmd-cli (3x faster)."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temp: float = 0.2,
        max_tokens: int = 256,
        threads: int = 0,
        verbose: bool = False,
        cli_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.temp = temp
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Find llama-mtmd-cli binary
        if cli_path and os.path.exists(cli_path):
            self.cli_path = cli_path
        else:
            self.cli_path = shutil.which("llama-mtmd-cli")
            if not self.cli_path:
                raise RuntimeError(
                    "llama-mtmd-cli not found in PATH. Install llama.cpp CLI tools or specify --cli-path"
                )
        LOG.info("Using CLI: %s", self.cli_path)

    def classify(
        self, img_path: str, meta: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify screenshot using CLI tool."""
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: Classify what the user is doing now. If matches any project by keywords, set project.name accordingly."

        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--image", img_path,
            "--temp", str(self.temp),
            "-n", str(self.max_tokens),
            "-p", full_prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"CLI failed with code {result.returncode}: {result.stderr}"
                )

            output = result.stdout
            LOG.debug("CLI output: %s", output[:500])
            obj = extract_json(output)
            return self._normalize(obj)

        except subprocess.TimeoutExpired:
            raise RuntimeError("CLI timeout after 120s")
        except Exception as e:
            LOG.error("CLI invocation failed: %r", e)
            raise

    def _normalize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate LLM output."""
        obj["coarse_activity"] = str(obj.get("coarse_activity", "misc")).lower()
        if obj["coarse_activity"] not in COARSE_ENUM:
            obj["coarse_activity"] = "misc"
        
        # Clamp confidences
        try:
            obj["confidence"] = float(max(0.0, min(1.0, obj.get("confidence", 0.0))))
        except Exception:
            obj["confidence"] = 0.0
        
        try:
            pj = obj.get("project") or {}
            pj["confidence"] = float(max(0.0, min(1.0, pj.get("confidence", 0.0))))
            obj["project"] = pj
        except Exception:
            obj["project"] = {"name": None, "confidence": 0.0, "reason": ""}
        
        return obj

"""Shared endpoint discovery and Ollama authentication for monitoring and authoring."""

import os

import httpx


def native_ollama(base_url):
    return httpx.URL(base_url).path.rstrip("/").endswith("/api")


def cloud_model(base_url, model):
    return httpx.URL(base_url).host == "ollama.com" or model.endswith((":cloud", "-cloud"))


def ollama_headers(base_url):
    url = httpx.URL(base_url)
    if url.host != "ollama.com":
        return {}
    if url.scheme != "https" or url.port not in (None, 443) or url.username or url.password:
        raise ValueError("Ollama Cloud requires https://ollama.com with no credentials in the URL")
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        raise ValueError("Set OLLAMA_API_KEY in the environment before starting RoboBrain")
    return {"Authorization": "Bearer " + key}


async def model_info(client, base_url, selected, timeout=8, *, require_vision=False):
    base_url, selected = base_url.rstrip("/"), selected.strip()
    headers = ollama_headers(base_url)
    if not native_ollama(base_url):
        response = await client.get(base_url + "/models", headers=headers, timeout=timeout)
        response.raise_for_status()
        if selected not in {m["id"] for m in response.json()["data"]}:
            raise ValueError("Configured model is not available at the endpoint")
        return {"model": selected, "api": "openai"}
    response = await client.get(base_url + "/tags", headers=headers, timeout=timeout)
    response.raise_for_status()
    names = {m["name"] for m in response.json()["models"]}
    model = selected if selected in names else selected.removesuffix(":cloud").removesuffix("-cloud")
    if model not in names:
        raise ValueError(f"Model {selected} is not available at the Ollama endpoint")
    response = await client.post(base_url + "/show", json={"model": model}, headers=headers, timeout=timeout)
    response.raise_for_status()
    capabilities = response.json().get("capabilities", [])
    if require_vision and "vision" not in capabilities:
        raise ValueError(f"Model {model} does not support image input required by the monitor")
    return {"model": model, "api": "ollama", "vision": "vision" in capabilities, "thinking": "thinking" in capabilities}


def endpoint_error(base_url, error):
    if httpx.URL(base_url).host == "ollama.com":
        advice = "Check OLLAMA_API_KEY, Ollama model access and cloud usage limits."
    elif native_ollama(base_url):
        advice = "Check that Ollama is running at the configured /api endpoint."
    else:
        advice = "Check the model server and its OpenAI-compatible endpoint; check the SSH forward if used."
    return f"Model endpoint unavailable: {error}. {advice}"

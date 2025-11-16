import openai


def expand_prompt(description, api_key, model="gpt-3.5-turbo", n=2):
    """
    Use OpenAI's GPT (>=1.0.0 API) to expand a descriptive phrase into multiple short prompts.

    Args:
        description (str): User-provided object description
        api_key (str): Your OpenAI API key
        model (str): OpenAI model name (default: "gpt-3.5-turbo")
        n (int): Number of prompt variants to generate (soft)

    Returns:
        List[str]: List of short prompts (e.g., "a small dog", "a brown dog", ...)
    """
    client = openai.OpenAI(api_key=api_key)

    system_prompt = (
        "You are a prompt rewriting assistant that helps convert a complex object description into multiple short natural phrases "
        "for object detection models like OWL-ViT. Keep phrases concise, object-focused, and diverse."
    )

    user_prompt = (
        f"Given the object description: \"{description}\", generate exactly {n} short phrases that describe the same object from different angles. "
        "Each phrase must:\n"
        "- Be 3 to 7 words long\n"
        "- Start with 'a' or 'an'\n"
        "- Describe the object only (not scene or action)\n"
        "- Be natural and suitable as an object detection prompt\n\n"
        "Output the phrases as plain text, one per line, with **no numbering, no dashes, and no extra text.**\n"
        "Example output:\n"
        "a small brown dog\n"
        "a puppy in a white shirt\n"
        "a cute animal\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content
        prompts = [line.strip("-• ").strip() for line in content.split("\n") if line.strip()]
        return list(set(prompts)) if prompts else [description]

    except Exception as e:
        print(f"[expand_prompt] OpenAI request failed: {e}")
        return [description]  # fallback

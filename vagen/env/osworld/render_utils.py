from .env_utils import ObsPostProcessor, encode_image, resize_image_from_bytes
import json


def compact(d, indent=0):
    def tight(obj):
        return json.dumps(obj, separators=(',', ':'))
    
    out_str = ''
    for i, (k, v) in enumerate(d.items()):
        comma = ',' if i < len(d) else ''
        out_str += f'{" " * indent}{tight(k)}:{tight(v)}{comma}\n'
    return out_str


def _process_raw_action_for_html(raw_response_str: str):
    special_word_replacement = {
        "<think>": "&lt;think&gt;",
        "</think>": "&lt;/think&gt;",
        "<action>": "&lt;action&gt;",
        "</action>": "&lt;/action&gt;",
        "<simulate>": "&lt;simulate&gt;",
        "</simulate>": "&lt;/simulate&gt;",
    }
    for k, v in special_word_replacement.items():
        raw_response_str = raw_response_str.replace(k, v)
    return raw_response_str


def render_train_trajectory_to_html(
    task_config: dict,
    trajectory: list,
    additional_actions: list[str],
    postprocesser: ObsPostProcessor,
    output_fpath: str
):
    instruction = task_config["instruction"]
    eval_config = task_config["evaluator"]
    eval_config_str = compact(eval_config, indent=4)

    content = f"<pre><em>Instruction:</em> {instruction}</pre>"
    content += f"<pre><em>Evaluator:</em><br/>{eval_config_str}</pre>"
    content += "<hr/>"
    
    additiona_act_str = ""
    for a_idx, action in enumerate(additional_actions):
        if a_idx % 2 == 0:
            additiona_act_str += f"<pre style='background-color: gray;'>{action}</pre>-----"
        else:
            additiona_act_str += f"<pre>{action}</pre>-----"
    if additiona_act_str:
        content += f"<pre><em>Additional actions:</em><br/>{additiona_act_str}</pre>"
        content += "<hr/>"
    
    for data in trajectory:
        if "obs" in data.keys():
            # is observation
            obs = data["obs"]
            processed_obs = postprocesser(obs)
            if postprocesser.observation_type in ["screenshot", "screenshot_a11y_tree"]:
                screenshot = obs['screenshot']
            else:
                screenshot = processed_obs['screenshot'] or obs['screenshot']
            ally_tree = processed_obs['accessibility_tree']

            screenshot = resize_image_from_bytes(screenshot, size=(960, 540))
            screenshot_b64 = encode_image(screenshot)

            content += (
                '<div class="obs">'
                    "<h4>Observation:</h4>"
                    f'<img src="data:image/png;base64,{screenshot_b64}"/>'
                    f'<pre>{ally_tree}</pre>'
                '</div>'
            )
        else:
            # is action
            raw_action = _process_raw_action_for_html(data["raw_action"])
            content += (
                '<div class="raw_action">'
                    '<h4>Raw Action:</h4>'
                    f'<pre>{raw_action}</pre>'
                '</div>'
            )
            content += (
                '<div class="action">'
                    f'<pre>{data["action"]}</pre>'
                '</div>'
            )
    
    style = (
        ".raw_action {background-color: grey;}\n"
        ".action {background-color: yellow;}\n"
        "pre {white-space: pre-wrap; word-wrap: break-word;}"
    )
    HTML_TEMPLATE = (
        "<html>\n"
        "<head>\n"
            "<style>\n"
                f"{style}\n"
            "</style>\n"
        "</head>\n"
            "<body>\n"
                f"{content}\n"
            "</body>\n"
        "</html>\n"
    )
    with open(output_fpath, "w", encoding="utf-8") as fwrite:
        fwrite.write(HTML_TEMPLATE)
    return
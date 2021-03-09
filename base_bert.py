import re
from torch import device, dtype
from config import BertConfig, PretrainedConfig
from utils import *


class BertPreTrainedModel(nn.Module):
  config_class = BertConfig
  base_model_prefix = "bert"
  _keys_to_ignore_on_load_missing = [r"position_ids"]
  _keys_to_ignore_on_load_unexpected = None

  def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    super().__init__()
    self.config = config
    self.name_or_path = config.name_or_path

  def init_weights(self):
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    mirror = kwargs.pop("mirror", None)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
      config_path = config if config is not None else pretrained_model_name_or_path
      config, model_kwargs = cls.config_class.from_pretrained(
        config_path,
        *model_args,
        cache_dir=cache_dir,
        return_unused_kwargs=True,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        local_files_only=local_files_only,
        use_auth_token=use_auth_token,
        revision=revision,
        **kwargs,
      )
    else:
      model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
      pretrained_model_name_or_path = str(pretrained_model_name_or_path)
      if os.path.isdir(pretrained_model_name_or_path):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
      elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
        archive_file = pretrained_model_name_or_path
      else:
        archive_file = hf_bucket_url(
          pretrained_model_name_or_path,
          filename=WEIGHTS_NAME,
          revision=revision,
          mirror=mirror,
        )
      try:
        # Load from URL or cache if already cached
        resolved_archive_file = cached_path(
          archive_file,
          cache_dir=cache_dir,
          force_download=force_download,
          proxies=proxies,
          resume_download=resume_download,
          local_files_only=local_files_only,
          use_auth_token=use_auth_token,
        )
      except EnvironmentError as err:
        #logger.error(err)
        msg = (
          f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
          f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
          f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}.\n\n"
        )
        raise EnvironmentError(msg)
    else:
      resolved_archive_file = None

    config.name_or_path = pretrained_model_name_or_path

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)

    if state_dict is None:
      try:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
      except Exception:
        raise OSError(
          f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
          f"at '{resolved_archive_file}'"
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    m = {'embeddings.word_embeddings': 'word_embedding',
         'embeddings.position_embeddings': 'pos_embedding',
         'embeddings.token_type_embeddings': 'tk_type_embedding',
         'embeddings.LayerNorm': 'embed_layer_norm',
         'embeddings.dropout': 'embed_dropout',
         'encoder.layer': 'bert_layers',
         'pooler.dense': 'pooler_dense',
         'pooler.activation': 'pooler_af',
         'attention.self': "self_attention",
         'attention.output.dense': 'attention_dense',
         'attention.output.LayerNorm': 'attention_layer_norm',
         'attention.output.dropout': 'attention_dropout',
         'intermediate.dense': 'interm_dense',
         'intermediate.intermediate_act_fn': 'interm_af',
         'output.dense': 'out_dense',
         'output.LayerNorm': 'out_layer_norm',
         'output.dropout': 'out_dropout'}

    for key in state_dict.keys():
      new_key = None
      if "gamma" in key:
        new_key = key.replace("gamma", "weight")
      if "beta" in key:
        new_key = key.replace("beta", "bias")
      for x, y in m.items():
        if new_key is not None:
          _key = new_key
        else:
          _key = key
        if x in key:
          new_key = _key.replace(x, y)
      if new_key:
        old_keys.append(key)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
      # print(old_key, new_key)
      state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
      state_dict._metadata = metadata

    your_bert_params = [f"bert.{x[0]}" for x in model.named_parameters()]
    for k in state_dict:
      if k not in your_bert_params and not k.startswith("cls."):
        possible_rename = [x for x in k.split(".")[1:-1] if x in m.values()]
        raise ValueError(f"{k} cannot be reload to your model, one/some of {possible_rename} we provided have been renamed")

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, prefix=""):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        True,
        missing_keys,
        unexpected_keys,
        error_msgs,
      )
      for name, child in module._modules.items():
        if child is not None:
          load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
    if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
      start_prefix = cls.base_model_prefix + "."
    if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
      model_to_load = getattr(model, cls.base_model_prefix)
    load(model_to_load, prefix=start_prefix)

    if model.__class__.__name__ != model_to_load.__class__.__name__:
      base_model_state_dict = model_to_load.state_dict().keys()
      head_model_state_dict_without_base_prefix = [
        key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
      ]
      missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if cls._keys_to_ignore_on_load_missing is not None:
      for pat in cls._keys_to_ignore_on_load_missing:
        missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
      for pat in cls._keys_to_ignore_on_load_unexpected:
        unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(error_msgs) > 0:
      raise RuntimeError(
        "Error(s) in loading state_dict for {}:\n\t{}".format(
          model.__class__.__name__, "\n\t".join(error_msgs)
        )
      )

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if output_loading_info:
      loading_info = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "error_msgs": error_msgs,
      }
      return model, loading_info

    if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
      import torch_xla.core.xla_model as xm

      model = xm.send_cpu_data_to_device(model, xm.xla_device())
      model.to(xm.xla_device())

    return model

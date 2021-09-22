from typing import List, Optional, Tuple, Dict, Union, Any, overload, Sequence, NamedTuple
import collections
import os
import re
import unicodedata
import itertools
import requests
import copy
import json
from contextlib import contextmanager
from collections import OrderedDict, UserDict
from enum import Enum
import numpy as np
from utils import cached_path, hf_bucket_url, is_remote_url, is_tf_available, is_torch_available
from tokenizers import AddedToken
from tokenizers import Encoding as EncodingFast


VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
FULL_TOKENIZER_FILE = "tokenizer.json"

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    }
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512
}
PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True}
}


TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class ExplicitEnum(Enum):
  @classmethod
  def _missing_(cls, value):
    raise ValueError(
      "%r is not a valid %s, please select one of %s"
      % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
    )


class TruncationStrategy(ExplicitEnum):
  ONLY_FIRST = "only_first"
  ONLY_SECOND = "only_second"
  LONGEST_FIRST = "longest_first"
  DO_NOT_TRUNCATE = "do_not_truncate"


class PaddingStrategy(ExplicitEnum):
  LONGEST = "longest"
  MAX_LENGTH = "max_length"
  DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
  PYTORCH = "pt"
  TENSORFLOW = "tf"
  NUMPY = "np"
  JAX = "jax"


class CharSpan(NamedTuple):
  start: int
  end: int


class TokenSpan(NamedTuple):
  start: int
  end: int


def to_py_obj(obj):
  """
  Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
  """
  if isinstance(obj, (dict, BatchEncoding)):
    return {k: to_py_obj(v) for k, v in obj.items()}
  elif isinstance(obj, (list, tuple)):
    return [to_py_obj(o) for o in obj]
  elif is_tf_available() and _is_tensorflow(obj):
    return obj.numpy().tolist()
  elif is_torch_available() and _is_torch(obj):
    return obj.detach().cpu().tolist()
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    return obj


def _is_torch(x):
  import torch
  return isinstance(x, torch.Tensor)


def _is_torch_device(x):
  import torch
  return isinstance(x, torch.device)


def _is_end_of_word(text):
  last_char = text[-1]
  return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
  first_char = text[0]
  return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _is_punctuation(char):
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def _is_whitespace(char):
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def load_vocab(vocab_file):
  vocab = collections.OrderedDict()
  with open(vocab_file, "r", encoding="utf-8") as reader:
    tokens = reader.readlines()
  for index, token in enumerate(tokens):
    token = token.rstrip("\n")
    vocab[token] = index
  return vocab


def whitespace_tokenize(text):
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class BatchEncoding(UserDict):
  def __init__(
    self,
    data: Optional[Dict[str, Any]] = None,
    encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
    tensor_type: Union[None, str, TensorType] = None,
    prepend_batch_axis: bool = False,
    n_sequences: Optional[int] = None,
  ):
    super().__init__(data)

    if isinstance(encoding, EncodingFast):
      encoding = [encoding]

    self._encodings = encoding

    if n_sequences is None and encoding is not None and len(encoding):
      n_sequences = encoding[0].n_sequences

    self._n_sequences = n_sequences

    self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

  @property
  def n_sequences(self) -> Optional[int]:
    return self._n_sequences

  @property
  def is_fast(self) -> bool:
    return self._encodings is not None

  def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
    if isinstance(item, str):
      return self.data[item]
    elif self._encodings is not None:
      return self._encodings[item]
    else:
      raise KeyError(
        "Indexing with integers (to access backend Encoding for a given batch index) "
        "is not available when using Python based tokenizers"
      )

  def __getattr__(self, item: str):
    try:
      return self.data[item]
    except KeyError:
      raise AttributeError

  def __getstate__(self):
    return {"data": self.data, "encodings": self._encodings}

  def __setstate__(self, state):
    if "data" in state:
      self.data = state["data"]

    if "encodings" in state:
      self._encodings = state["encodings"]

  def keys(self):
    return self.data.keys()

  def values(self):
    return self.data.values()

  def items(self):
    return self.data.items()

  # After this point:
  # Extended properties and methods only available for fast (Rust-based) tokenizers
  # provided by HuggingFace tokenizers library.

  @property
  def encodings(self) -> Optional[List[EncodingFast]]:
    return self._encodings

  def tokens(self, batch_index: int = 0) -> List[str]:
    if not self._encodings:
      raise ValueError("tokens() is not available when using Python-based tokenizers")
    return self._encodings[batch_index].tokens

  def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
    if not self._encodings:
      raise ValueError("sequence_ids() is not available when using Python-based tokenizers")
    return self._encodings[batch_index].sequence_ids

  def words(self, batch_index: int = 0) -> List[Optional[int]]:
    if not self._encodings:
      raise ValueError("words() is not available when using Python-based tokenizers")
    return self.word_ids(batch_index)

  def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
    if not self._encodings:
      raise ValueError("word_ids() is not available when using Python-based tokenizers")
    return self._encodings[batch_index].word_ids

  def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
    if not self._encodings:
      raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
    if token_index is not None:
      batch_index = batch_or_token_index
    else:
      batch_index = 0
      token_index = batch_or_token_index
    if batch_index < 0:
      batch_index = self._batch_size + batch_index
    if token_index < 0:
      token_index = self._seq_len + token_index
    return self._encodings[batch_index].token_to_sequence(token_index)

  def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
    if not self._encodings:
      raise ValueError("token_to_word() is not available when using Python based tokenizers")
    if token_index is not None:
      batch_index = batch_or_token_index
    else:
      batch_index = 0
      token_index = batch_or_token_index
    if batch_index < 0:
      batch_index = self._batch_size + batch_index
    if token_index < 0:
      token_index = self._seq_len + token_index
    return self._encodings[batch_index].token_to_word(token_index)

  def word_to_tokens(
    self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
  ) -> Optional[TokenSpan]:
    if not self._encodings:
      raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
    if word_index is not None:
      batch_index = batch_or_word_index
    else:
      batch_index = 0
      word_index = batch_or_word_index
    if batch_index < 0:
      batch_index = self._batch_size + batch_index
    if word_index < 0:
      word_index = self._seq_len + word_index
    span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
    return TokenSpan(*span) if span is not None else None

  def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
    if not self._encodings:
      raise ValueError("token_to_chars() is not available when using Python based tokenizers")
    if token_index is not None:
      batch_index = batch_or_token_index
    else:
      batch_index = 0
      token_index = batch_or_token_index
    return CharSpan(*(self._encodings[batch_index].token_to_chars(token_index)))

  def char_to_token(
    self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
  ) -> int:
    if not self._encodings:
      raise ValueError("char_to_token() is not available when using Python based tokenizers")
    if char_index is not None:
      batch_index = batch_or_char_index
    else:
      batch_index = 0
      char_index = batch_or_char_index
    return self._encodings[batch_index].char_to_token(char_index, sequence_index)

  def word_to_chars(
    self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
  ) -> CharSpan:
    if not self._encodings:
      raise ValueError("word_to_chars() is not available when using Python based tokenizers")
    if word_index is not None:
      batch_index = batch_or_word_index
    else:
      batch_index = 0
      word_index = batch_or_word_index
    return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))

  def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
    if not self._encodings:
      raise ValueError("char_to_word() is not available when using Python based tokenizers")
    if char_index is not None:
      batch_index = batch_or_char_index
    else:
      batch_index = 0
      char_index = batch_or_char_index
    return self._encodings[batch_index].char_to_word(char_index, sequence_index)

  def convert_to_tensors(
    self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
  ):
    if tensor_type is None:
      return self

    # Convert to TensorType
    if not isinstance(tensor_type, TensorType):
      tensor_type = TensorType(tensor_type)

    # Get a function reference for the correct framework
    if tensor_type == TensorType.TENSORFLOW:
      if not is_tf_available():
        raise ImportError(
          "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
        )
      import tensorflow as tf

      as_tensor = tf.constant
      is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
      if not is_torch_available():
        raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
      import torch

      as_tensor = torch.tensor
      is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
      if not is_flax_available():
        raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
      import jax.numpy as jnp  # noqa: F811

      as_tensor = jnp.array
      is_tensor = _is_jax
    else:
      as_tensor = np.asarray
      is_tensor = _is_numpy
    # (mfuntowicz: This code is unreachable)
    # else:
    #     raise ImportError(
    #         "Unable to convert output to tensors format {}".format(tensor_type)
    #     )

    # Do the tensor conversion in batch
    for key, value in self.items():
      try:
        if prepend_batch_axis:
          value = [value]

        if not is_tensor(value):
          tensor = as_tensor(value)

          # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
          # # at-least2d
          # if tensor.ndim > 2:
          #     tensor = tensor.squeeze(0)
          # elif tensor.ndim < 2:
          #     tensor = tensor[None, :]

          self[key] = tensor
      except:  # noqa E722
        if key == "overflowing_tokens":
          raise ValueError(
            "Unable to create tensor returning overflowing tokens of different lengths. "
            "Please see if a fast version of this tokenizer is available to have this feature available."
          )
        raise ValueError(
          "Unable to create tensor, you should probably activate truncation and/or padding "
          "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
        )

    return self

  def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
    # This check catches things like APEX blindly calling "to" on all inputs to a module
    # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
    # into a HalfTensor
    if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
      self.data = {k: v.to(device=device) for k, v in self.data.items()}
    return self


class SpecialTokensMixin:
  SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
  ]

  def __init__(self, verbose=True, **kwargs):
    self._bos_token = None
    self._eos_token = None
    self._unk_token = None
    self._sep_token = None
    self._pad_token = None
    self._cls_token = None
    self._mask_token = None
    self._pad_token_type_id = 0
    self._additional_special_tokens = []
    self.verbose = verbose

    # We directly set the hidden value to allow initialization with special tokens
    # which are not yet in the vocabulary. Necessary for serialization/de-serialization
    # TODO clean this up at some point (probably by switching to fast tokenizers)
    for key, value in kwargs.items():
      if value is None:
        continue
      if key in self.SPECIAL_TOKENS_ATTRIBUTES:
        if key == "additional_special_tokens":
          assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
          assert all(isinstance(t, str) for t in value), "One of the tokens is not a string"
          setattr(self, key, value)
        elif isinstance(value, (str, AddedToken)):
          setattr(self, key, value)
        else:
          raise TypeError(
            "special token {} has to be either str or AddedToken but got: {}".format(key, type(value))
          )

  def sanitize_special_tokens(self) -> int:
    return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

  def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]) -> int:
    if not special_tokens_dict:
      return 0

    added_tokens = 0
    for key, value in special_tokens_dict.items():
      assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

      setattr(self, key, value)

      if key == "additional_special_tokens":
        assert isinstance(value, (list, tuple)) and all(
          isinstance(t, (str, AddedToken)) for t in value
        ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
        added_tokens += self.add_tokens(value, special_tokens=True)
      else:
        assert isinstance(
          value, (str, AddedToken)
        ), f"Token {value} for key {key} should be a str or an AddedToken instance"
        added_tokens += self.add_tokens([value], special_tokens=True)

    return added_tokens

  def add_tokens(
    self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
  ) -> int:
    if not new_tokens:
      return 0

    if not isinstance(new_tokens, (list, tuple)):
      new_tokens = [new_tokens]

    return self._add_tokens(new_tokens, special_tokens=special_tokens)

  def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
    raise NotImplementedError

  @property
  def bos_token(self) -> str:
    if self._bos_token is None and self.verbose:
      return None
    return str(self._bos_token)

  @property
  def eos_token(self) -> str:
    if self._eos_token is None and self.verbose:
      return None
    return str(self._eos_token)

  @property
  def unk_token(self) -> str:
    if self._unk_token is None and self.verbose:
      return None
    return str(self._unk_token)

  @property
  def sep_token(self) -> str:
    if self._sep_token is None and self.verbose:
      return None
    return str(self._sep_token)

  @property
  def pad_token(self) -> str:
    if self._pad_token is None and self.verbose:
      return None
    return str(self._pad_token)

  @property
  def cls_token(self) -> str:
    if self._cls_token is None and self.verbose:
      return None
    return str(self._cls_token)

  @property
  def mask_token(self) -> str:
    if self._mask_token is None and self.verbose:
      return None
    return str(self._mask_token)

  @property
  def additional_special_tokens(self) -> List[str]:
    if self._additional_special_tokens is None and self.verbose:
      return None
    return [str(tok) for tok in self._additional_special_tokens]

  @bos_token.setter
  def bos_token(self, value):
    self._bos_token = value

  @eos_token.setter
  def eos_token(self, value):
    self._eos_token = value

  @unk_token.setter
  def unk_token(self, value):
    self._unk_token = value

  @sep_token.setter
  def sep_token(self, value):
    self._sep_token = value

  @pad_token.setter
  def pad_token(self, value):
    self._pad_token = value

  @cls_token.setter
  def cls_token(self, value):
    self._cls_token = value

  @mask_token.setter
  def mask_token(self, value):
    self._mask_token = value

  @additional_special_tokens.setter
  def additional_special_tokens(self, value):
    self._additional_special_tokens = value

  @property
  def bos_token_id(self) -> Optional[int]:
    if self._bos_token is None:
      return None
    return self.convert_tokens_to_ids(self.bos_token)

  @property
  def eos_token_id(self) -> Optional[int]:
    if self._eos_token is None:
      return None
    return self.convert_tokens_to_ids(self.eos_token)

  @property
  def unk_token_id(self) -> Optional[int]:
    if self._unk_token is None:
      return None
    return self.convert_tokens_to_ids(self.unk_token)

  @property
  def sep_token_id(self) -> Optional[int]:
    if self._sep_token is None:
      return None
    return self.convert_tokens_to_ids(self.sep_token)

  @property
  def pad_token_id(self) -> Optional[int]:
    if self._pad_token is None:
      return None
    return self.convert_tokens_to_ids(self.pad_token)

  @property
  def pad_token_type_id(self) -> int:
    return self._pad_token_type_id

  @property
  def cls_token_id(self) -> Optional[int]:
    if self._cls_token is None:
      return None
    return self.convert_tokens_to_ids(self.cls_token)

  @property
  def mask_token_id(self) -> Optional[int]:
    if self._mask_token is None:
      return None
    return self.convert_tokens_to_ids(self.mask_token)

  @property
  def additional_special_tokens_ids(self) -> List[int]:
    return self.convert_tokens_to_ids(self.additional_special_tokens)

  @bos_token_id.setter
  def bos_token_id(self, value):
    self._bos_token = self.convert_tokens_to_ids(value)

  @eos_token_id.setter
  def eos_token_id(self, value):
    self._eos_token = self.convert_tokens_to_ids(value)

  @unk_token_id.setter
  def unk_token_id(self, value):
    self._unk_token = self.convert_tokens_to_ids(value)

  @sep_token_id.setter
  def sep_token_id(self, value):
    self._sep_token = self.convert_tokens_to_ids(value)

  @pad_token_id.setter
  def pad_token_id(self, value):
    self._pad_token = self.convert_tokens_to_ids(value)

  @cls_token_id.setter
  def cls_token_id(self, value):
    self._cls_token = self.convert_tokens_to_ids(value)

  @mask_token_id.setter
  def mask_token_id(self, value):
    self._mask_token = self.convert_tokens_to_ids(value)

  @additional_special_tokens_ids.setter
  def additional_special_tokens_ids(self, values):
    self._additional_special_tokens = [self.convert_tokens_to_ids(value) for value in values]

  @property
  def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
    set_attr = {}
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
      attr_value = getattr(self, "_" + attr)
      if attr_value:
        set_attr[attr] = str(attr_value)
    return set_attr

  @property
  def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
    set_attr = {}
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
      attr_value = getattr(self, "_" + attr)
      if attr_value:
        set_attr[attr] = attr_value
    return set_attr

  @property
  def all_special_tokens(self) -> List[str]:
    all_toks = [str(s) for s in self.all_special_tokens_extended]
    return all_toks

  @property
  def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
    all_toks = []
    set_attr = self.special_tokens_map_extended
    for attr_value in set_attr.values():
      all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
    all_toks = list(OrderedDict.fromkeys(all_toks))
    return all_toks

  @property
  def all_special_ids(self) -> List[int]:
    all_toks = self.all_special_tokens
    all_ids = self.convert_tokens_to_ids(all_toks)
    return all_ids


class PreTrainedTokenizerBase(SpecialTokensMixin):
  vocab_files_names: Dict[str, str] = {}
  pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
  pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
  max_model_input_sizes: Dict[str, Optional[int]] = {}

  # first name has to correspond to main model input name
  # to make sure `tokenizer.pad(...)` works correctly
  model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
  padding_side: str = "right"
  slow_tokenizer_class = None

  def __init__(self, **kwargs):
    # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
    self.init_inputs = ()
    self.init_kwargs = copy.deepcopy(kwargs)
    self.name_or_path = kwargs.pop("name_or_path", "")

    # For backward compatibility we fallback to set model_max_length from max_len if provided
    model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
    self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

    # Padding side is right by default and overridden in subclasses. If specified in the kwargs, it is changed.
    self.padding_side = kwargs.pop("padding_side", self.padding_side)
    assert self.padding_side in [
      "right",
      "left",
    ], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
    self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

    self.deprecation_warnings = (
      {}
    )  # Use to store when we have already noticed a deprecation warning (avoid overlogging).

    super().__init__(**kwargs)

  @property
  def max_len_single_sentence(self) -> int:
    return self.model_max_length - self.num_special_tokens_to_add(pair=False)

  @property
  def max_len_sentences_pair(self) -> int:
    return self.model_max_length - self.num_special_tokens_to_add(pair=True)

  @max_len_single_sentence.setter
  def max_len_single_sentence(self, value) -> int:
    # For backward compatibility, allow to try to setup 'max_len_single_sentence'.
    if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
      self.deprecation_warnings["max_len_single_sentence"] = True
    else:
      raise ValueError(
        "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
      )

  @max_len_sentences_pair.setter
  def max_len_sentences_pair(self, value) -> int:
    # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
    if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
      self.deprecation_warnings["max_len_sentences_pair"] = True
    else:
      raise ValueError(
        "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
      )

  def __repr__(self) -> str:
    return (
      f"{'PreTrainedTokenizerFast' if self.is_fast else 'PreTrainedTokenizer'}(name_or_path='{self.name_or_path}', "
      f"vocab_size={self.vocab_size}, model_max_len={self.model_max_length}, is_fast={self.is_fast}, "
      f"padding_side='{self.padding_side}', special_tokens={self.special_tokens_map_extended})"
    )

  def get_vocab(self) -> Dict[str, int]:
    raise NotImplementedError()

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)

    s3_models = list(cls.max_model_input_sizes.keys())
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    vocab_files = {}
    init_configuration = {}
    if pretrained_model_name_or_path in s3_models:
      # Get the vocabulary from AWS S3 bucket
      for file_id, map_list in cls.pretrained_vocab_files_map.items():
        vocab_files[file_id] = map_list[pretrained_model_name_or_path]
      if (
        cls.pretrained_init_configuration
        and pretrained_model_name_or_path in cls.pretrained_init_configuration
      ):
        init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path].copy()
    else:
      # Get the vocabulary from local files
      if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
        if len(cls.vocab_files_names) > 1:
          raise ValueError(
            "Calling {}.from_pretrained() with the path to a single file or url is not supported."
            "Use a model identifier or the path to a directory instead.".format(cls.__name__)
          )
        file_id = list(cls.vocab_files_names.keys())[0]
        vocab_files[file_id] = pretrained_model_name_or_path
      else:
        # At this point pretrained_model_name_or_path is either a directory or a model identifier name
        additional_files_names = {
          "added_tokens_file": ADDED_TOKENS_FILE,
          "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
          "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
          "tokenizer_file": FULL_TOKENIZER_FILE,
        }
        # Look for the tokenizer files
        for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
          if os.path.isdir(pretrained_model_name_or_path):
            if subfolder is not None:
              full_file_name = os.path.join(pretrained_model_name_or_path, subfolder, file_name)
            else:
              full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
            if not os.path.exists(full_file_name):
              full_file_name = None
          else:
            full_file_name = hf_bucket_url(
              pretrained_model_name_or_path,
              filename=file_name,
              subfolder=subfolder,
              revision=revision,
              mirror=None,
            )

          vocab_files[file_id] = full_file_name

    # Get files from url, cache, or disk depending on the case
    resolved_vocab_files = {}
    unresolved_files = []
    for file_id, file_path in vocab_files.items():
      if file_path is None:
        resolved_vocab_files[file_id] = None
      else:
        try:
          try:
            resolved_vocab_files[file_id] = cached_path(
              file_path,
              cache_dir=cache_dir,
              force_download=force_download,
              proxies=proxies,
              resume_download=resume_download,
              local_files_only=local_files_only,
              use_auth_token=use_auth_token,
            )
          except FileNotFoundError as error:
            if local_files_only:
              unresolved_files.append(file_id)
            else:
              raise error

        except requests.exceptions.HTTPError as err:
          if "404 Client Error" in str(err):
            resolved_vocab_files[file_id] = None
          else:
            raise err

    if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
      msg = (
        f"Can't load tokenizer for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing relevant tokenizer files\n\n"
      )
      raise EnvironmentError(msg)

    for file_id, file_path in vocab_files.items():
      if file_id not in resolved_vocab_files:
        continue

    return cls._from_pretrained(
      resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs
    )

  @classmethod
  def _from_pretrained(
    cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs
  ):
    # We instantiate fast tokenizers based on a slow tokenizer if we don't have access to the tokenizer.json
    # file or if `from_slow` is set to True.
    from_slow = kwargs.get("from_slow", False)
    has_tokenizer_file = resolved_vocab_files.get("tokenizer_file", None) is not None
    if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None:
      slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
        copy.deepcopy(resolved_vocab_files),
        pretrained_model_name_or_path,
        copy.deepcopy(init_configuration),
        *init_inputs,
        **(copy.deepcopy(kwargs)),
      )
    else:
      slow_tokenizer = None

    # Prepare tokenizer initialization kwargs
    # Did we saved some inputs and kwargs to reload ?
    tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
    if tokenizer_config_file is not None:
      with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
        init_kwargs = json.load(tokenizer_config_handle)
      saved_init_inputs = init_kwargs.pop("init_inputs", ())
      if not init_inputs:
        init_inputs = saved_init_inputs
    else:
      init_kwargs = init_configuration

    # Update with newly provided kwargs
    init_kwargs.update(kwargs)

    # Convert AddedTokens serialized as dict to class instances
    def convert_added_tokens(obj: Union[AddedToken, Any]):
      if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
        obj.pop("__type")
        return AddedToken(**obj)
      elif isinstance(obj, (list, tuple)):
        return list(convert_added_tokens(o) for o in obj)
      elif isinstance(obj, dict):
        return {k: convert_added_tokens(v) for k, v in obj.items()}
      return obj

    init_kwargs = convert_added_tokens(init_kwargs)

    # Set max length if needed
    if pretrained_model_name_or_path in cls.max_model_input_sizes:
      # if we're using a pretrained model, ensure the tokenizer
      # wont index sequences longer than the number of positional embeddings
      model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
      if model_max_length is not None and isinstance(model_max_length, (int, float)):
        init_kwargs["model_max_length"] = min(init_kwargs.get("model_max_length", int(1e30)), model_max_length)

    # Merge resolved_vocab_files arguments in init_kwargs.
    added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
    for args_name, file_path in resolved_vocab_files.items():
      if args_name not in init_kwargs:
        init_kwargs[args_name] = file_path

    if slow_tokenizer is not None:
      init_kwargs["__slow_tokenizer"] = slow_tokenizer

    init_kwargs["name_or_path"] = pretrained_model_name_or_path

    # Instantiate tokenizer.
    try:
      tokenizer = cls(*init_inputs, **init_kwargs)
    except OSError:
      raise OSError(
        "Unable to load vocabulary from file. "
        "Please check that the provided vocabulary is accessible and not corrupted."
      )

    # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
    # Removed: Now done at the base class level
    # tokenizer.init_inputs = init_inputs
    # tokenizer.init_kwargs = init_kwargs

    # If there is a complementary special token map, load it
    special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
    if special_tokens_map_file is not None:
      with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
        special_tokens_map = json.load(special_tokens_map_handle)
      for key, value in special_tokens_map.items():
        if isinstance(value, dict):
          value = AddedToken(**value)
        elif isinstance(value, list):
          value = [AddedToken(**token) if isinstance(token, dict) else token for token in value]
        setattr(tokenizer, key, value)

    # Add supplementary tokens.
    special_tokens = tokenizer.all_special_tokens
    if added_tokens_file is not None:
      with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
        added_tok_encoder = json.load(added_tokens_handle)

      # Sort added tokens by index
      added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(), key=lambda x: x[1]))

      for token, index in added_tok_encoder_sorted:
        assert index == len(tokenizer), (
          f"Non-consecutive added token '{token}' found. "
          f"Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
        )
        tokenizer.add_tokens(token, special_tokens=bool(token in special_tokens))

    # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
    added_tokens = tokenizer.sanitize_special_tokens()

    return tokenizer

  def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    legacy_format: bool = True,
    filename_prefix: Optional[str] = None,
  ) -> Tuple[str]:
    if os.path.isfile(save_directory):
      return
    os.makedirs(save_directory, exist_ok=True)

    special_tokens_map_file = os.path.join(
      save_directory, (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE
    )
    tokenizer_config_file = os.path.join(
      save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
    )

    tokenizer_config = copy.deepcopy(self.init_kwargs)
    if len(self.init_inputs) > 0:
      tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
    for file_id in self.vocab_files_names.keys():
      tokenizer_config.pop(file_id, None)

    # Sanitize AddedTokens
    def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
      if isinstance(obj, AddedToken):
        out = obj.__getstate__()
        if add_type_field:
          out["__type"] = "AddedToken"
        return out
      elif isinstance(obj, (list, tuple)):
        return list(convert_added_tokens(o, add_type_field=add_type_field) for o in obj)
      elif isinstance(obj, dict):
        return {k: convert_added_tokens(v, add_type_field=add_type_field) for k, v in obj.items()}
      return obj

    # add_type_field=True to allow dicts in the kwargs / differentiate from AddedToken serialization
    tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)
    with open(tokenizer_config_file, "w", encoding="utf-8") as f:
      f.write(json.dumps(tokenizer_config, ensure_ascii=False))

    # Sanitize AddedTokens in special_tokens_map
    write_dict = convert_added_tokens(self.special_tokens_map_extended, add_type_field=False)
    with open(special_tokens_map_file, "w", encoding="utf-8") as f:
      f.write(json.dumps(write_dict, ensure_ascii=False))

    file_names = (tokenizer_config_file, special_tokens_map_file)

    return self._save_pretrained(
      save_directory=save_directory,
      file_names=file_names,
      legacy_format=legacy_format,
      filename_prefix=filename_prefix,
    )

  def _save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    file_names: Tuple[str],
    legacy_format: bool = True,
    filename_prefix: Optional[str] = None,
  ) -> Tuple[str]:
    if not legacy_format:
      raise ValueError(
        "Only fast tokenizers (instances of PretrainedTokenizerFast) can be saved in non legacy format."
      )

    save_directory = str(save_directory)

    added_tokens_file = os.path.join(
      save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
    )
    added_vocab = self.get_added_vocab()
    if added_vocab:
      with open(added_tokens_file, "w", encoding="utf-8") as f:
        out_str = json.dumps(added_vocab, ensure_ascii=False)
        f.write(out_str)

    vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

    return file_names + vocab_files + (added_tokens_file,)

  def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    raise NotImplementedError

  def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
    raise NotImplementedError

  def encode(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    stride: int = 0,
    return_tensors: Optional[Union[str, TensorType]] = None,
    **kwargs
  ) -> List[int]:
    encoded_inputs = self.encode_plus(
      text,
      text_pair=text_pair,
      add_special_tokens=add_special_tokens,
      padding=padding,
      truncation=truncation,
      max_length=max_length,
      stride=stride,
      return_tensors=return_tensors,
      **kwargs,
    )

    return encoded_inputs["input_ids"]

  def num_special_tokens_to_add(self, pair: bool = False) -> int:
    raise NotImplementedError

  def _get_padding_truncation_strategies(
    self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
  ):
    old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
    old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

    # Backward compatibility for previous behavior, maybe we should deprecate it:
    # If you only set max_length, it activates truncation for max_length
    if max_length is not None and padding is False and truncation is False:
      if verbose:
        self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
      truncation = "longest_first"

    # Get padding strategy
    if padding is False and old_pad_to_max_length:
      if max_length is None:
        padding_strategy = PaddingStrategy.LONGEST
      else:
        padding_strategy = PaddingStrategy.MAX_LENGTH
    elif padding is not False:
      if padding is True:
        padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
      elif not isinstance(padding, PaddingStrategy):
        padding_strategy = PaddingStrategy(padding)
      elif isinstance(padding, PaddingStrategy):
        padding_strategy = padding
    else:
      padding_strategy = PaddingStrategy.DO_NOT_PAD

    # Get truncation strategy
    if truncation is False and old_truncation_strategy != "do_not_truncate":
      truncation_strategy = TruncationStrategy(old_truncation_strategy)
    elif truncation is not False:
      if truncation is True:
        truncation_strategy = (
          TruncationStrategy.LONGEST_FIRST
        )  # Default to truncate the longest sequences in pairs of inputs
      elif not isinstance(truncation, TruncationStrategy):
        truncation_strategy = TruncationStrategy(truncation)
      elif isinstance(truncation, TruncationStrategy):
        truncation_strategy = truncation
    else:
      truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

    # Set max length if needed
    if max_length is None:
      if padding_strategy == PaddingStrategy.MAX_LENGTH:
        if self.model_max_length > LARGE_INTEGER:
          if verbose:
            self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
          padding_strategy = PaddingStrategy.DO_NOT_PAD
        else:
          max_length = self.model_max_length

      if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
        if self.model_max_length > LARGE_INTEGER:
          if verbose:
            self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
          truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
        else:
          max_length = self.model_max_length

    # Test if we have a padding token
    if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad_token or self.pad_token_id < 0):
      raise ValueError(
        "Asking to pad but the tokenizer does not have a padding token. "
        "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
        "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
      )

    # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
    if (
      truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
      and padding_strategy != PaddingStrategy.DO_NOT_PAD
      and pad_to_multiple_of is not None
      and max_length is not None
      and (max_length % pad_to_multiple_of != 0)
    ):
      raise ValueError(
        f"Truncation and padding are both activated but "
        f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
      )

    return padding_strategy, truncation_strategy, max_length, kwargs

  def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    # Input type checking for clearer error
    assert isinstance(text, str) or (
      isinstance(text, (list, tuple))
      and (
        len(text) == 0
        or (
          isinstance(text[0], str)
          or (isinstance(text[0], (list, tuple)) and (len(text[0]) == 0 or isinstance(text[0][0], str)))
        )
      )
    ), (
      "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
      "or `List[List[str]]` (batch of pretokenized examples)."
    )

    assert (
      text_pair is None
      or isinstance(text_pair, str)
      or (
        isinstance(text_pair, (list, tuple))
        and (
          len(text_pair) == 0
          or (
            isinstance(text_pair[0], str)
            or (
              isinstance(text_pair[0], (list, tuple))
              and (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str))
            )
          )
        )
      )
    ), (
      "text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
      "or `List[List[str]]` (batch of pretokenized examples)."
    )

    is_batched = bool(
      (not is_split_into_words and isinstance(text, (list, tuple)))
      or (
        is_split_into_words and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
      )
    )

    if is_batched:
      batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
      return self.batch_encode_plus(
        batch_text_or_text_pairs=batch_text_or_text_pairs,
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
        **kwargs,
      )
    else:
      return self.encode_plus(
        text=text,
        text_pair=text_pair,
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
        **kwargs,
      )

  def encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
    padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
      padding=padding,
      truncation=truncation,
      max_length=max_length,
      pad_to_multiple_of=pad_to_multiple_of,
      verbose=verbose,
      **kwargs,
    )

    return self._encode_plus(
      text=text,
      text_pair=text_pair,
      add_special_tokens=add_special_tokens,
      padding_strategy=padding_strategy,
      truncation_strategy=truncation_strategy,
      max_length=max_length,
      stride=stride,
      is_split_into_words=is_split_into_words,
      pad_to_multiple_of=pad_to_multiple_of,
      return_tensors=return_tensors,
      return_token_type_ids=return_token_type_ids,
      return_attention_mask=return_attention_mask,
      return_overflowing_tokens=return_overflowing_tokens,
      return_special_tokens_mask=return_special_tokens_mask,
      return_offsets_mapping=return_offsets_mapping,
      return_length=return_length,
      verbose=verbose,
      **kwargs,
    )

  def _encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    raise NotImplementedError

  def batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
      List[TextInput],
      List[TextInputPair],
      List[PreTokenizedInput],
      List[PreTokenizedInputPair],
      List[EncodedInput],
      List[EncodedInputPair],
    ],
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
    padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
      padding=padding,
      truncation=truncation,
      max_length=max_length,
      pad_to_multiple_of=pad_to_multiple_of,
      verbose=verbose,
      **kwargs,
    )

    return self._batch_encode_plus(
      batch_text_or_text_pairs=batch_text_or_text_pairs,
      add_special_tokens=add_special_tokens,
      padding_strategy=padding_strategy,
      truncation_strategy=truncation_strategy,
      max_length=max_length,
      stride=stride,
      is_split_into_words=is_split_into_words,
      pad_to_multiple_of=pad_to_multiple_of,
      return_tensors=return_tensors,
      return_token_type_ids=return_token_type_ids,
      return_attention_mask=return_attention_mask,
      return_overflowing_tokens=return_overflowing_tokens,
      return_special_tokens_mask=return_special_tokens_mask,
      return_offsets_mapping=return_offsets_mapping,
      return_length=return_length,
      verbose=verbose,
      **kwargs,
    )

  def _batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
      List[TextInput],
      List[TextInputPair],
      List[PreTokenizedInput],
      List[PreTokenizedInputPair],
      List[EncodedInput],
      List[EncodedInputPair],
    ],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    raise NotImplementedError

  def pad(
    self,
    encoded_inputs: Union[
      BatchEncoding,
      List[BatchEncoding],
      Dict[str, EncodedInput],
      Dict[str, List[EncodedInput]],
      List[Dict[str, EncodedInput]],
    ],
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    verbose: bool = True,
  ) -> BatchEncoding:
    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
      encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

    # The model's main input name, usually `input_ids`, has be passed for padding
    if self.model_input_names[0] not in encoded_inputs:
      raise ValueError(
        "You should supply an encoding or a list of encodings to this method"
        f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
      )

    required_input = encoded_inputs[self.model_input_names[0]]

    if not required_input:
      if return_attention_mask:
        encoded_inputs["attention_mask"] = []
      return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
      # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
      index = 0
      while len(required_input[index]) == 0:
        index += 1
      if index < len(required_input):
        first_element = required_input[index][0]
    # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
    if not isinstance(first_element, (int, list, tuple)):
      if is_tf_available() and _is_tensorflow(first_element):
        return_tensors = "tf" if return_tensors is None else return_tensors
      elif is_torch_available() and _is_torch(first_element):
        return_tensors = "pt" if return_tensors is None else return_tensors
      elif isinstance(first_element, np.ndarray):
        return_tensors = "np" if return_tensors is None else return_tensors
      else:
        raise ValueError(
          f"type of {first_element} unknown: {type(first_element)}. "
          f"Should be one of a python, numpy, pytorch or tensorflow object."
        )

      for key, value in encoded_inputs.items():
        encoded_inputs[key] = to_py_obj(value)

    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
      padding=padding, max_length=max_length, verbose=verbose
    )

    required_input = encoded_inputs[self.model_input_names[0]]
    if required_input and not isinstance(required_input[0], (list, tuple)):
      encoded_inputs = self._pad(
        encoded_inputs,
        max_length=max_length,
        padding_strategy=padding_strategy,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
      )
      return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    batch_size = len(required_input)
    assert all(
      len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    if padding_strategy == PaddingStrategy.LONGEST:
      max_length = max(len(inputs) for inputs in required_input)
      padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
      inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
      outputs = self._pad(
        inputs,
        max_length=max_length,
        padding_strategy=padding_strategy,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
      )

      for key, value in outputs.items():
        if key not in batch_outputs:
          batch_outputs[key] = []
        batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)

  def create_token_type_ids_from_sequences(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
  ) -> List[int]:
    if token_ids_1 is None:
      return len(token_ids_0) * [0]
    return [0] * len(token_ids_0) + [1] * len(token_ids_1)

  def build_inputs_with_special_tokens(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
  ) -> List[int]:
    if token_ids_1 is None:
      return token_ids_0
    return token_ids_0 + token_ids_1

  def prepare_for_model(
    self,
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    stride: int = 0,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    prepend_batch_axis: bool = False,
    **kwargs
  ) -> BatchEncoding:
    # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
    padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
      padding=padding,
      truncation=truncation,
      max_length=max_length,
      pad_to_multiple_of=pad_to_multiple_of,
      verbose=verbose,
      **kwargs,
    )

    pair = bool(pair_ids is not None)
    len_ids = len(ids)
    len_pair_ids = len(pair_ids) if pair else 0

    if return_token_type_ids and not add_special_tokens:
      raise ValueError(
        "Asking to return token_type_ids while setting add_special_tokens to False "
        "results in an undefined behavior. Please set add_special_tokens to True or "
        "set return_token_type_ids to None."
      )

    # Load from model defaults
    if return_token_type_ids is None:
      return_token_type_ids = "token_type_ids" in self.model_input_names
    if return_attention_mask is None:
      return_attention_mask = "attention_mask" in self.model_input_names

    encoded_inputs = {}

    # Compute the total size of the returned encodings
    total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

    # Truncation: Handle max sequence length
    overflowing_tokens = []
    if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
      ids, pair_ids, overflowing_tokens = self.truncate_sequences(
        ids,
        pair_ids=pair_ids,
        num_tokens_to_remove=total_len - max_length,
        truncation_strategy=truncation_strategy,
        stride=stride,
      )

    if return_overflowing_tokens:
      encoded_inputs["overflowing_tokens"] = overflowing_tokens
      encoded_inputs["num_truncated_tokens"] = total_len - max_length

    # Add special tokens
    if add_special_tokens:
      sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
      token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
    else:
      sequence = ids + pair_ids if pair else ids
      token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

    # Build output dictionary
    encoded_inputs["input_ids"] = sequence
    if return_token_type_ids:
      encoded_inputs["token_type_ids"] = token_type_ids
    if return_special_tokens_mask:
      if add_special_tokens:
        encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
      else:
        encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

    # Check lengths
    self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

    # Padding
    if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
      encoded_inputs = self.pad(
        encoded_inputs,
        max_length=max_length,
        padding=padding_strategy.value,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
      )

    if return_length:
      encoded_inputs["length"] = len(encoded_inputs["input_ids"])

    batch_outputs = BatchEncoding(
      encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
    )

    return batch_outputs

  def truncate_sequences(
    self,
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    num_tokens_to_remove: int = 0,
    truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
    stride: int = 0,
  ) -> Tuple[List[int], List[int], List[int]]:
    if num_tokens_to_remove <= 0:
      return ids, pair_ids, []

    if not isinstance(truncation_strategy, TruncationStrategy):
      truncation_strategy = TruncationStrategy(truncation_strategy)

    overflowing_tokens = []
    if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
      for _ in range(num_tokens_to_remove):
        if pair_ids is None or len(ids) > len(pair_ids):
          if not overflowing_tokens:
            window_len = min(len(ids), stride + 1)
          else:
            window_len = 1
          overflowing_tokens.extend(ids[-window_len:])
          ids = ids[:-1]
        else:
          if not overflowing_tokens:
            window_len = min(len(pair_ids), stride + 1)
          else:
            window_len = 1
          overflowing_tokens.extend(pair_ids[-window_len:])
          pair_ids = pair_ids[:-1]
    elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
      if len(ids) > num_tokens_to_remove:
        window_len = min(len(ids), stride + num_tokens_to_remove)
        overflowing_tokens = ids[-window_len:]
        ids = ids[:-num_tokens_to_remove]
    elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
      if len(pair_ids) > num_tokens_to_remove:
        window_len = min(len(pair_ids), stride + num_tokens_to_remove)
        overflowing_tokens = pair_ids[-window_len:]
        pair_ids = pair_ids[:-num_tokens_to_remove]

    return (ids, pair_ids, overflowing_tokens)

  def _pad(
    self,
    encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
  ) -> dict:
    # Load from model defaults
    if return_attention_mask is None:
      return_attention_mask = "attention_mask" in self.model_input_names

    required_input = encoded_inputs[self.model_input_names[0]]

    if padding_strategy == PaddingStrategy.LONGEST:
      max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
      max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

    if needs_to_be_padded:
      difference = max_length - len(required_input)
      if self.padding_side == "right":
        if return_attention_mask:
          encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
        if "token_type_ids" in encoded_inputs:
          encoded_inputs["token_type_ids"] = (
            encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
          )
        if "special_tokens_mask" in encoded_inputs:
          encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
        encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
      elif self.padding_side == "left":
        if return_attention_mask:
          encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
        if "token_type_ids" in encoded_inputs:
          encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
            "token_type_ids"
          ]
        if "special_tokens_mask" in encoded_inputs:
          encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
        encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
      else:
        raise ValueError("Invalid padding strategy:" + str(self.padding_side))
    elif return_attention_mask and "attention_mask" not in encoded_inputs:
      encoded_inputs["attention_mask"] = [1] * len(required_input)

    return encoded_inputs

  def convert_tokens_to_string(self, tokens: List[str]) -> str:
    raise NotImplementedError

  def batch_decode(
    self,
    sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
  ) -> List[str]:
    return [
      self.decode(
        seq,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs,
      )
      for seq in sequences
    ]

  def decode(
    self,
    token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
  ) -> str:
    # Convert inputs to python lists
    token_ids = to_py_obj(token_ids)

    return self._decode(
      token_ids=token_ids,
      skip_special_tokens=skip_special_tokens,
      clean_up_tokenization_spaces=clean_up_tokenization_spaces,
      **kwargs,
    )

  def _decode(
    self,
    token_ids: Union[int, List[int]],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
  ) -> str:
    raise NotImplementedError

  def get_special_tokens_mask(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
  ) -> List[int]:
    assert already_has_special_tokens and token_ids_1 is None, (
      "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
      "Please use a slow (full python) tokenizer to activate this argument."
      "Or set `return_special_tokens_mask=True` when calling the encoding method "
      "to get the special tokens mask in any tokenizer. "
    )

    all_special_ids = self.all_special_ids  # cache the property

    special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

    return special_tokens_mask

  @staticmethod
  def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
      out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

  def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
    if max_length is None and len(ids) > self.model_max_length and verbose:
      self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

  @contextmanager
  def as_target_tokenizer(self):
    yield

  def prepare_seq2seq_batch(
    self,
    src_texts: List[str],
    tgt_texts: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
    padding: str = "longest",
    return_tensors: str = None,
    truncation: bool = True,
    **kwargs,
  ) -> BatchEncoding:
    # mBART-specific kwargs that should be ignored by other models.
    kwargs.pop("src_lang", None)
    kwargs.pop("tgt_lang", None)
    if max_length is None:
      max_length = self.model_max_length
    model_inputs = self(
      src_texts,
      add_special_tokens=True,
      return_tensors=return_tensors,
      max_length=max_length,
      padding=padding,
      truncation=truncation,
      **kwargs,
    )
    if tgt_texts is None:
      return model_inputs
    # Process tgt_texts
    if max_target_length is None:
      max_target_length = max_length
    with self.as_target_tokenizer():
      labels = self(
        tgt_texts,
        add_special_tokens=True,
        return_tensors=return_tensors,
        padding=padding,
        max_length=max_target_length,
        truncation=truncation,
        **kwargs,
      )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class PreTrainedTokenizer(PreTrainedTokenizerBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Added tokens - We store this for both slow and fast tokenizers
    # until the serialization of Fast tokenizers is updated
    self.added_tokens_encoder: Dict[str, int] = {}
    self.added_tokens_decoder: Dict[int, str] = {}
    self.unique_no_split_tokens: List[str] = []

  @property
  def is_fast(self) -> bool:
    return False

  @property
  def vocab_size(self) -> int:
    """
    :obj:`int`: Size of the base vocabulary (without the added tokens).
    """
    raise NotImplementedError

  def get_added_vocab(self) -> Dict[str, int]:
    """
    Returns the added tokens in the vocabulary as a dictionary of token to index.
    Returns:
        :obj:`Dict[str, int]`: The added tokens.
    """
    return self.added_tokens_encoder

  def __len__(self):
    """
    Size of the full vocabulary with the added tokens.
    """
    return self.vocab_size + len(self.added_tokens_encoder)

  def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
    """
    Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
    it with indices starting from length of the current vocabulary.
    Args:
        new_tokens (:obj:`List[str]`or :obj:`List[tokenizers.AddedToken]`):
            Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
            checking if the tokenizer assign the index of the ``unk_token`` to them).
        special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the tokens should be added as special tokens.
    Returns:
        :obj:`int`: The number of tokens actually added to the vocabulary.
    Examples::
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
        print('We have added', num_added_toks, 'tokens')
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
    """
    new_tokens = [str(tok) for tok in new_tokens]

    tokens_to_add = []
    for token in new_tokens:
      assert isinstance(token, str)
      if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
        token = token.lower()
      if (
        token != self.unk_token
        and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
        and token not in tokens_to_add
      ):
        tokens_to_add.append(token)

    added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
    added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
    self.added_tokens_encoder.update(added_tok_encoder)
    self.added_tokens_decoder.update(added_tok_decoder)

    # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
    if special_tokens:
      self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
    else:
      # Or on the newly added tokens
      self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))

    return len(tokens_to_add)

  def num_special_tokens_to_add(self, pair: bool = False) -> int:
    """
    Returns the number of added tokens when encoding a sequence with special tokens.
    .. note::
        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not
        put this inside your training loop.
    Args:
        pair (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the number of added tokens should be computed in the case of a sequence pair or a single
            sequence.
    Returns:
        :obj:`int`: Number of special tokens added to sequences.
    """
    token_ids_0 = []
    token_ids_1 = []
    return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

  def tokenize(self, text: TextInput, **kwargs) -> List[str]:
    """
    Converts a string in a sequence of tokens, using the tokenizer.
    Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
    (BPE/SentencePieces/WordPieces). Takes care of added tokens.
    Args:
        text (:obj:`str`):
            The sequence to be encoded.
        **kwargs (additional keyword arguments):
            Passed along to the model-specific ``prepare_for_tokenization`` preprocessing method.
    Returns:
        :obj:`List[str]`: The list of tokens.
    """
    # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
    all_special_tokens_extended = dict(
      (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
    )

    text, kwargs = self.prepare_for_tokenization(text, **kwargs)

    # TODO: should this be in the base class?
    if hasattr(self, "do_lower_case") and self.do_lower_case:
      # convert non-special tokens to lowercase
      escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
      pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
      text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    def split_on_token(tok, text):
      result = []
      tok_extended = all_special_tokens_extended.get(tok, None)
      split_text = text.split(tok)
      full_word = ""
      for i, sub_text in enumerate(split_text):
        # AddedToken can control whitespace stripping around them.
        # We use them for GPT2 and Roberta to have different behavior depending on the special token
        # Cf. https://github.com/huggingface/transformers/pull/2778
        # and https://github.com/huggingface/transformers/issues/3788
        if isinstance(tok_extended, AddedToken):
          if tok_extended.single_word:
            # Try to avoid splitting on token
            if (
              i < len(split_text) - 1
              and not _is_end_of_word(sub_text)
              and not _is_start_of_word(split_text[i + 1])
            ):
              # Don't extract the special token
              full_word += sub_text + tok
            elif full_word:
              full_word += sub_text
              result.append(full_word)
              full_word = ""
              continue
          # Strip white spaces on the right
          if tok_extended.rstrip and i > 0:
            # A bit counter-intuitive but we strip the left of the string
            # since tok_extended.rstrip means the special token is eating all white spaces on its right
            sub_text = sub_text.lstrip()
          # Strip white spaces on the left
          if tok_extended.lstrip and i < len(split_text) - 1:
            sub_text = sub_text.rstrip()  # Opposite here
        else:
          # We strip left and right by default
          if i < len(split_text) - 1:
            sub_text = sub_text.rstrip()
          if i > 0:
            sub_text = sub_text.lstrip()

        if i == 0 and not sub_text:
          result.append(tok)
        elif i == len(split_text) - 1:
          if sub_text:
            result.append(sub_text)
          else:
            pass
        else:
          if sub_text:
            result.append(sub_text)
          result.append(tok)
      return result

    def split_on_tokens(tok_list, text):
      if not text.strip():
        return []
      if not tok_list:
        return self._tokenize(text)

      tokenized_text = []
      text_list = [text]
      for tok in tok_list:
        tokenized_text = []
        for sub_text in text_list:
          if sub_text not in self.unique_no_split_tokens:
            tokenized_text.extend(split_on_token(tok, sub_text))
          else:
            tokenized_text.append(sub_text)
        text_list = tokenized_text

      return list(
        itertools.chain.from_iterable(
          (
            self._tokenize(token) if token not in self.unique_no_split_tokens else [token]
            for token in tokenized_text
          )
        )
      )

    no_split_token = self.unique_no_split_tokens
    tokenized_text = split_on_tokens(no_split_token, text)
    return tokenized_text

  def _tokenize(self, text, **kwargs):
    """
    Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
    vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).
    Do NOT take care of added tokens.
    """
    raise NotImplementedError

  def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
    vocabulary.
    Args:
        tokens (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).
    Returns:
        :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
    """
    if tokens is None:
      return None

    if isinstance(tokens, str):
      return self._convert_token_to_id_with_added_voc(tokens)

    ids = []
    for token in tokens:
      ids.append(self._convert_token_to_id_with_added_voc(token))
    return ids

  def _convert_token_to_id_with_added_voc(self, token):
    if token is None:
      return None

    if token in self.added_tokens_encoder:
      return self.added_tokens_encoder[token]
    return self._convert_token_to_id(token)

  def _convert_token_to_id(self, token):
    raise NotImplementedError

  def _encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    def get_input_ids(text):
      if isinstance(text, str):
        tokens = self.tokenize(text, **kwargs)
        return self.convert_tokens_to_ids(tokens)
      elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
        if is_split_into_words:
          tokens = list(
            itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
          )
          return self.convert_tokens_to_ids(tokens)
        else:
          return self.convert_tokens_to_ids(text)
      elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
        return text
      else:
        if is_split_into_words:
          raise ValueError(
            f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
          )
        else:
          raise ValueError(
            f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
          )

    if return_offsets_mapping:
      raise NotImplementedError(
        "return_offset_mapping is not available when using Python tokenizers."
        "To use this feature, change your tokenizer to one deriving from "
        "transformers.PreTrainedTokenizerFast."
        "More information on available tokenizers at "
        "https://github.com/huggingface/transformers/pull/2674"
      )

    first_ids = get_input_ids(text)
    second_ids = get_input_ids(text_pair) if text_pair is not None else None

    return self.prepare_for_model(
      first_ids,
      pair_ids=second_ids,
      add_special_tokens=add_special_tokens,
      padding=padding_strategy.value,
      truncation=truncation_strategy.value,
      max_length=max_length,
      stride=stride,
      pad_to_multiple_of=pad_to_multiple_of,
      return_tensors=return_tensors,
      prepend_batch_axis=True,
      return_attention_mask=return_attention_mask,
      return_token_type_ids=return_token_type_ids,
      return_overflowing_tokens=return_overflowing_tokens,
      return_special_tokens_mask=return_special_tokens_mask,
      return_length=return_length,
      verbose=verbose,
    )

  def _batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
      List[TextInput],
      List[TextInputPair],
      List[PreTokenizedInput],
      List[PreTokenizedInputPair],
      List[EncodedInput],
      List[EncodedInputPair],
    ],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs
  ) -> BatchEncoding:
    def get_input_ids(text):
      if isinstance(text, str):
        tokens = self.tokenize(text, **kwargs)
        return self.convert_tokens_to_ids(tokens)
      elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
        if is_split_into_words:
          tokens = list(
            itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
          )
          return self.convert_tokens_to_ids(tokens)
        else:
          return self.convert_tokens_to_ids(text)
      elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
        return text
      else:
        raise ValueError(
          "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
        )

    if return_offsets_mapping:
      raise NotImplementedError(
        "return_offset_mapping is not available when using Python tokenizers."
        "To use this feature, change your tokenizer to one deriving from "
        "transformers.PreTrainedTokenizerFast."
      )

    input_ids = []
    for ids_or_pair_ids in batch_text_or_text_pairs:
      if not isinstance(ids_or_pair_ids, (list, tuple)):
        ids, pair_ids = ids_or_pair_ids, None
      elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
        ids, pair_ids = ids_or_pair_ids, None
      else:
        ids, pair_ids = ids_or_pair_ids

      first_ids = get_input_ids(ids)
      second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
      input_ids.append((first_ids, second_ids))

    batch_outputs = self._batch_prepare_for_model(
      input_ids,
      add_special_tokens=add_special_tokens,
      padding_strategy=padding_strategy,
      truncation_strategy=truncation_strategy,
      max_length=max_length,
      stride=stride,
      pad_to_multiple_of=pad_to_multiple_of,
      return_attention_mask=return_attention_mask,
      return_token_type_ids=return_token_type_ids,
      return_overflowing_tokens=return_overflowing_tokens,
      return_special_tokens_mask=return_special_tokens_mask,
      return_length=return_length,
      return_tensors=return_tensors,
      verbose=verbose,
    )

    return BatchEncoding(batch_outputs)

  def _batch_prepare_for_model(
    self,
    batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_length: bool = False,
    verbose: bool = True,
  ) -> BatchEncoding:
    """
    Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
    adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
    manages a moving window (with user defined stride) for overflowing tokens
    Args:
        batch_ids_pairs: list of tokenized input ids or input ids pairs
    """

    batch_outputs = {}
    for first_ids, second_ids in batch_ids_pairs:
      outputs = self.prepare_for_model(
        first_ids,
        second_ids,
        add_special_tokens=add_special_tokens,
        padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
        truncation=truncation_strategy.value,
        max_length=max_length,
        stride=stride,
        pad_to_multiple_of=None,  # we pad in batch afterward
        return_attention_mask=False,  # we pad in batch afterward
        return_token_type_ids=return_token_type_ids,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_length=return_length,
        return_tensors=None,  # We convert the whole batch to tensors at the end
        prepend_batch_axis=False,
        verbose=verbose,
      )

      for key, value in outputs.items():
        if key not in batch_outputs:
          batch_outputs[key] = []
        batch_outputs[key].append(value)

    batch_outputs = self.pad(
      batch_outputs,
      padding=padding_strategy.value,
      max_length=max_length,
      pad_to_multiple_of=pad_to_multiple_of,
      return_attention_mask=return_attention_mask,
    )

    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

    return batch_outputs

  def prepare_for_tokenization(
    self, text: str, is_split_into_words: bool = False, **kwargs
  ) -> Tuple[str, Dict[str, Any]]:
    """
    Performs any necessary transformations before tokenization.
    This method should pop the arguments from kwargs and return the remaining :obj:`kwargs` as well. We test the
    :obj:`kwargs` at the end of the encoding process to be sure all the arguments have been used.
    Args:
        text (:obj:`str`):
            The text to prepare.
        is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the text has been pretokenized.
        kwargs:
            Keyword arguments to use for the tokenization.
    Returns:
        :obj:`Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
    """
    return (text, kwargs)

  def get_special_tokens_mask(
    self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
  ) -> List[int]:
    """
    Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
    special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
    Args:
        token_ids_0 (:obj:`List[int]`):
            List of ids of the first sequence.
        token_ids_1 (:obj:`List[int]`, `optional`):
            List of ids of the second sequence.
        already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the token list is already formatted with special tokens for the model.
    Returns:
        A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    """
    return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

  @overload
  def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
    ...

  @overload
  def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
    ...

  def convert_ids_to_tokens(
    self, ids: Union[int, List[int]], skip_special_tokens: bool = False
  ) -> Union[str, List[str]]:
    """
    Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
    added tokens.
    Args:
        ids (:obj:`int` or :obj:`List[int]`):
            The token id (or token ids) to convert to tokens.
        skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to remove special tokens in the decoding.
    Returns:
        :obj:`str` or :obj:`List[str]`: The decoded token(s).
    """
    if isinstance(ids, int):
      if ids in self.added_tokens_decoder:
        return self.added_tokens_decoder[ids]
      else:
        return self._convert_id_to_token(ids)
    tokens = []
    for index in ids:
      index = int(index)
      if skip_special_tokens and index in self.all_special_ids:
        continue
      if index in self.added_tokens_decoder:
        tokens.append(self.added_tokens_decoder[index])
      else:
        tokens.append(self._convert_id_to_token(index))
    return tokens

  def _convert_id_to_token(self, index: int) -> str:
    raise NotImplementedError

  def convert_tokens_to_string(self, tokens: List[str]) -> str:
    return " ".join(tokens)

  def _decode(
    self,
    token_ids: List[int],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = True,
    spaces_between_special_tokens: bool = True,
  ) -> str:
    filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

    # To avoid mixing byte-level and unicode for byte-level BPT
    # we need to build string separately for added tokens and byte-level tokens
    # cf. https://github.com/huggingface/transformers/issues/1133
    sub_texts = []
    current_sub_text = []
    for token in filtered_tokens:
      if skip_special_tokens and token in self.all_special_ids:
        continue
      if token in self.added_tokens_encoder:
        if current_sub_text:
          sub_texts.append(self.convert_tokens_to_string(current_sub_text))
          current_sub_text = []
        sub_texts.append(token)
      else:
        current_sub_text.append(token)
    if current_sub_text:
      sub_texts.append(self.convert_tokens_to_string(current_sub_text))

    if spaces_between_special_tokens:
      text = " ".join(sub_texts)
    else:
      text = "".join(sub_texts)

    if clean_up_tokenization_spaces:
      clean_text = self.clean_up_tokenization(text)
      return clean_text
    else:
      return text



class BertTokenizer(PreTrainedTokenizer):
  vocab_files_names = VOCAB_FILES_NAMES
  pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
  pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
  max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

  def __init__(
    self,
    vocab_file,
    do_lower_case=True,
    do_basic_tokenize=True,
    never_split=None,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]",
    tokenize_chinese_chars=True,
    strip_accents=None,
    **kwargs
  ):
    super().__init__(
      do_lower_case=do_lower_case,
      do_basic_tokenize=do_basic_tokenize,
      never_split=never_split,
      unk_token=unk_token,
      sep_token=sep_token,
      pad_token=pad_token,
      cls_token=cls_token,
      mask_token=mask_token,
      tokenize_chinese_chars=tokenize_chinese_chars,
      strip_accents=strip_accents,
      **kwargs,
    )
    self.vocab = load_vocab(vocab_file)
    self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
    self.do_basic_tokenize = do_basic_tokenize
    if do_basic_tokenize:
      self.basic_tokenizer = BasicTokenizer(
        do_lower_case=do_lower_case,
        never_split=never_split,
        tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents,
      )
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

  @property
  def do_lower_case(self):
    return self.basic_tokenizer.do_lower_case

  @property
  def vocab_size(self):
    return len(self.vocab)

  def get_vocab(self):
    return dict(self.vocab, **self.added_tokens_encoder)

  def _tokenize(self, text):
    split_tokens = []
    if self.do_basic_tokenize:
      for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

        # If the token is part of the never_split set
        if token in self.basic_tokenizer.never_split:
          split_tokens.append(token)
        else:
          split_tokens += self.wordpiece_tokenizer.tokenize(token)
    else:
      split_tokens = self.wordpiece_tokenizer.tokenize(text)
    return split_tokens

  def _convert_token_to_id(self, token):
    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index):
    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens):
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

  def build_inputs_with_special_tokens(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
  ) -> List[int]:
    if token_ids_1 is None:
      return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    cls = [self.cls_token_id]
    sep = [self.sep_token_id]
    return cls + token_ids_0 + sep + token_ids_1 + sep

  def get_special_tokens_mask(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
  ) -> List[int]:
    if already_has_special_tokens:
      if token_ids_1 is not None:
        raise ValueError(
          "You should not supply a second sequence if the provided sequence of "
          "ids is already formatted with special tokens for the model."
        )
      return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

    if token_ids_1 is not None:
      return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    return [1] + ([0] * len(token_ids_0)) + [1]

  def create_token_type_ids_from_sequences(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
  ) -> List[int]:
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]
    if token_ids_1 is None:
      return len(cls + token_ids_0 + sep) * [0]
    return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

  def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    index = 0
    if os.path.isdir(save_directory):
      vocab_file = os.path.join(
        save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
      )
    else:
      vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
        if index != token_index:
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)


class BasicTokenizer(object):
  def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
    if never_split is None:
      never_split = []
    self.do_lower_case = do_lower_case
    self.never_split = set(never_split)
    self.tokenize_chinese_chars = tokenize_chinese_chars
    self.strip_accents = strip_accents

  def tokenize(self, text, never_split=None):
    # union() returns a new set by concatenating the two sets.
    never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    if self.tokenize_chinese_chars:
      text = self._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if token not in never_split:
        if self.do_lower_case:
          token = token.lower()
          if self.strip_accents is not False:
            token = self._run_strip_accents(token)
        elif self.strip_accents:
          token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token, never_split))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text, never_split=None):
    if never_split is not None and text in never_split:
      return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
      (cp >= 0x4E00 and cp <= 0x9FFF)
      or (cp >= 0x3400 and cp <= 0x4DBF)  #
      or (cp >= 0x20000 and cp <= 0x2A6DF)  #
      or (cp >= 0x2A700 and cp <= 0x2B73F)  #
      or (cp >= 0x2B740 and cp <= 0x2B81F)  #
      or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
      or (cp >= 0xF900 and cp <= 0xFAFF)
      or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
      return True

    return False

  def _clean_text(self, text):
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xFFFD or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens

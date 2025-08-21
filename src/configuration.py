import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional

import dataconf


class ConfigurationBase:
    @staticmethod
    def _convert_private_value(value: str):
        return value.replace('"#', '"pswd_')

    @staticmethod
    def _convert_private_value_inv(value: str):
        if value and value.startswith("pswd_"):
            return value.replace("pswd_", "#", 1)
        else:
            return value

    @classmethod
    def load_from_dict(cls, configuration: dict):
        """
        Initialize the configuration dataclass object from dictionary.
        Args:
            configuration: Dictionary loaded from json configuration.

        Returns:

        """
        json_conf = json.dumps(configuration)
        json_conf = ConfigurationBase._convert_private_value(json_conf)
        return dataconf.loads(json_conf, cls, ignore_unexpected=True)

    @classmethod
    def get_dataclass_required_parameters(cls) -> List[str]:
        """
        Return list of required parameters based on the dataclass definition (no default value)
        Returns: List[str]

        """
        return [
            cls._convert_private_value_inv(f.name)
            for f in dataclasses.fields(cls)
            if f.default == dataclasses.MISSING and f.default_factory == dataclasses.MISSING
        ]


@dataclass
class Destination(ConfigurationBase):
    incremental_load: bool
    output_table_name: str
    primary_keys_array: list[str]


@dataclass
class AdditionalOptions(ConfigurationBase):
    top_p: float
    max_tokens: int
    temperature: float
    presence_penalty: float
    frequency_penalty: float
    timeout: int = 60
    reasoning_effort: Optional[str] = None


@dataclass
class Authentication(ConfigurationBase):
    service: str
    pswd_api_token: str
    api_base: str = ""
    deployment_id: str = ""
    api_version: str = ""


@dataclass
class PromptOptions(ConfigurationBase):
    prompt: str


@dataclass
class PromptTemplates(ConfigurationBase):
    prompt_template: str


@dataclass
class Configuration(ConfigurationBase):
    destination: Destination
    additional_options: AdditionalOptions
    authentication: Authentication
    model: str
    endpoint_url: str = ""
    debug: bool = False
    max_token_spend: int = 0
    prompt_templates: PromptTemplates = field(default_factory=lambda: PromptTemplates(prompt_template=""))
    prompt_options: PromptOptions = field(default_factory=lambda: PromptOptions(prompt=""))

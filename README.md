Generative AI
=============

This component enables you to query OpenAI, Azure OpenAI and Llama 2 (coming soon) with data provided from your KBC project.


[TOC]


Configuration
=============

Accepts following parameters:


AI Service Provider **OpenAI**

- API Key `#api_token` You can get your API key in [OpenAI platform settings](https://platform.openai.com/account/api-keys)

AI Service Provider **Azure OpenAI**

You can find out how to get your API key, API Base and Deployment ID in [Microsoft Azure documentation](https://learn.microsoft.com/cs-cz/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python#retrieve-key-and-endpoint)
- API Key `#api_token`
- API Base `api_base`
- Deployment ID `deployment_id`
- API Version `api_version` API version use to call Completions endpoint with. You can see the list of supported API version in [Microsoft Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference?WT.mc_id=AZ-MVP-5004796#completions)

- Prompt `prompt` The prompt and data input pattern. Refer to input column using placeholder [[INPUT_COLUMN]]. The input table must contain the referenced column.
- Incremental Load `incremental load` If incremental load is turned on, the table will be updated instead of rewritten. Tables with a primary key will have rows updated, tables without a primary key will have rows appended.
- Output Table name `output_table_name` Name of the table stored in Storage.
- Primary Keys array `primary_keys_array` You can enter multiple columns separated by commas at once e.g. id, other_id. If a primary key is set, updates can be done on the table by selecting incremental loads. The primary key can consist of multiple columns. The primary key of an existing table cannot be changed.
- Predefined model `predefined_model` The model which will generate the completion. [Learn more](https://beta.openai.com/docs/models).

Additional options:

- Max Tokens `max_tokens` The maximum number of tokens to generate in the completion.\n\nThe token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
- Temperature `temperature` What sampling temperature to use [0-1]. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
- Top P `top_p` An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
- Frequency Penalty `frequency_penalty` Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
- Presence Penalty `presence_penalty` Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.


### Component Configuration

**Generic configuration**

```json
{
  "parameters": {
    "#api_token": "secret_api_token",
    "sleep": 5
  }
}
```

**Row configuration**

```json
{
  "parameters": {
    "prompt": "Extract keywords from this text:\n\n\"\"\"\n[[INPUT]]\n\"\"\"",
    "model_type": "predefined",
    "destination": {
      "incremental_load": true,
      "output_table_name": "keywords_test",
      "primary_keys_array": [
        "message_id"
      ]
    },
    "predefined_model": "text-davinci-002",
    "additional_options": {
      "top_p": 1,
      "max_tokens": 100,
      "temperature": 0.6,
      "presence_penalty": 0,
      "frequency_penalty": 0
    }
  }
}
```


# Development

If required, change local data folder (the `CUSTOM_FOLDER` placeholder) path to your custom path in
the `docker-compose.yml` file:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volumes:
      - ./:/code
      - ./CUSTOM_FOLDER:/data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone this repository, init the workspace and run the component with following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone git@bitbucket.org:kds_consulting_team/kds_consulting_team/kds-team.app-open-ai.git kds-team.app-open-ai
cd kds-team.app-open-ai
docker-compose build
docker-compose run --rm dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the test suite and lint check using this command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
docker-compose run --rm test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration
===========

For information about deployment and integration with KBC, please refer to the
[deployment section of developers documentation](https://developers.keboola.com/extend/component/deployment/)

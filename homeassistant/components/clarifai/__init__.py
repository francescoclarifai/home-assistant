"""The Clarifai integration."""
from clarifai.rest import ClarifaiApp
import voluptuous as vol

from homeassistant.const import CONF_API_KEY
import homeassistant.helpers.config_validation as cv

DOMAIN = "clarifai"
CONF_MODEL_NAME = "model_name"

DATA_CLARIFAI_MODEL = "clarifai_model"

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Required(CONF_API_KEY): cv.string,
                vol.Optional(CONF_MODEL_NAME, default="general"): cv.string,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass, config):
    """Set up the Clarifai component."""
    if DOMAIN not in config:
        return True

    client = ClarifaiApp(api_key=config[DOMAIN].get(CONF_API_KEY))
    model = client.models.get(config[DOMAIN].get(CONF_MODEL_NAME))

    hass.data[DATA_CLARIFAI_MODEL] = model

    return True

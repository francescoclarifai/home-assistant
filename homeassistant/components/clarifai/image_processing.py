"""Image processing with Clarifai model."""
from clarifai.rest import ClarifaiApp
import voluptuous as vol

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_SOURCE,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

DOMAIN = "clarifai"
CONF_MODEL_NAME = "model_name"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_MODEL_NAME, default="general"): cv.string,
    }
)


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Clarifai model platform."""

    client = ClarifaiApp(api_key=config[CONF_API_KEY])
    model = client.models.get(config[CONF_MODEL_NAME])

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            ClarifaiClassificationEntity(
                camera[CONF_ENTITY_ID], model, config[CONF_MODEL_NAME]
            )
        )

    async_add_entities(entities)


class ClarifaiClassificationEntity(ImageProcessingEntity):
    """Clarifai classification entity."""

    def __init__(self, camera_entity, model, name=None):
        """Initialize entity."""
        super().__init__()
        self._predictions = {}
        self._model = model
        self._camera = camera_entity
        self._state = None

        self._name = "Clarifai {}, camera {}".format(
            name, split_entity_id(camera_entity)[1]
        )

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the entity."""
        return self._name

    async def async_process_image(self, image):
        """Process image."""
        response = self._model.predict_by_bytes(bytearray(image))
        predictions = {}
        if "concepts" in response["outputs"][0]["data"].keys():  # classifier
            results = response["outputs"][0]["data"]["concepts"]
            for concept in results[:5]:
                predictions[concept["name"]] = round(concept["value"], 2)
        elif "regions" in response["outputs"][0]["data"].keys():  # detector
            results = response["outputs"][0]["data"]["regions"]
            for ii, region in enumerate(results):
                predictions["detection {}".format(ii)] = region["data"]["concepts"][0][
                    "name"
                ]
        else:
            pass

        self._predictions = predictions
        self._state = 1

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        attr["predictions"] = self._predictions
        return attr

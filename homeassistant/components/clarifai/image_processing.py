"""Image processing with Clarifai model."""
from homeassistant.components.clarifai import DATA_CLARIFAI_MODEL
from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    ImageProcessingEntity,
)
from homeassistant.core import split_entity_id


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Clarifai model platform."""
    model = hass.data[DATA_CLARIFAI_MODEL]

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            ClarifaiClassificationEntity(
                camera[CONF_ENTITY_ID], model, camera.get(CONF_NAME)
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

        if name:
            self._name = name
        else:
            self._name = "Clarifai {0}".format(split_entity_id(camera_entity)[1])

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
        response = self._model.predict_by_bytes(
            bytearray(image), min_value=0.5, max_concepts=5
        )
        predictions = {}
        results = response["outputs"][0]["data"].get("concepts", [])
        for concept in results:
            predictions[concept["name"]] = concept["value"]
        self._predictions = predictions
        self._state = 1

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        attr["predictions"] = self._predictions
        return attr

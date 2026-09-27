"""Backend-independent image request validation (decoding, order, limits).

llama.cpp compilation, orchestration and engine fidelity live in
test_llama_vision.py; this module keeps the upstream validation checks.
"""
import base64
import copy
import io

import pytest

from hf_vision import image_messages


def data_url(color='red'):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), color).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def payload():
    return {'model': 'test', 'messages': [
        {'role': 'system', 'content': 'Inspect the supplied pictures.'},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'First picture:'},
            {'type': 'image_url', 'image_url': {'url': data_url()}},
        ]},
        {'role': 'assistant', 'content': 'I will inspect it.'},
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': data_url('blue')}},
            {'type': 'text', 'text': 'Compare with this picture.'},
        ]},
    ], 'questions': {
        'color': {'type': 'choice', 'instructions': 'First color?', 'criteria': {'red': None, 'blue': None}},
        'red': {'type': 'noul', 'instructions': 'Is the first picture red?'},
        'level': {'type': 'score', 'instructions': 'How red is the first?', 'criteria': ['not red', 'red']},
    }}


def test_decode_preserves_order_and_request():
    request = payload()
    original = copy.deepcopy(request)
    messages, images = image_messages(request['messages'])
    assert request == original
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 0, 255)
    assert messages[1]['content'][1] == {'type': 'image'}


@pytest.mark.parametrize('url', ['https://example.org/image.png', 'file:///etc/passwd',
                                 'data:image/png;base64,@@@', 'data:image/png;base64,YQ==',
                                 'data:video/mp4;base64,YQ=='])
def test_reject_invalid_images(url):
    with pytest.raises(ValueError):
        image_messages([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}])


def test_camera_exif_orientation():
    from PIL import Image
    image = Image.new('RGB', (24, 16), 'red')
    exif = Image.Exif()
    exif[274] = 6  # camera orientation: rotate 90 degrees clockwise
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    url = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()
    _, images = image_messages([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}},
    ]}])
    assert images[0].size == (16, 24)
    assert images[0].getexif().get(274) is None


def test_image_limits(monkeypatch):
    import hf_vision
    request = payload()
    monkeypatch.setattr(hf_vision, 'MAX_IMAGES', 1)
    with pytest.raises(ValueError, match='count'):
        image_messages(request['messages'])
    monkeypatch.setattr(hf_vision, 'MAX_IMAGES', 16)
    monkeypatch.setattr(hf_vision, 'MAX_IMAGE_PIXELS', 10)
    with pytest.raises(ValueError, match='pixel'):
        image_messages(request['messages'])

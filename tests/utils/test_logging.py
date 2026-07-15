from roll.utils.logging import format_model_summary


class ModelWithBrokenRepr:
    def __repr__(self):
        raise AssertionError("model repr must not be called")


def test_format_model_summary_does_not_call_model_repr():
    models = [ModelWithBrokenRepr(), ModelWithBrokenRepr()]

    assert format_model_summary(models) == "2 model chunk(s): ModelWithBrokenRepr, ModelWithBrokenRepr"

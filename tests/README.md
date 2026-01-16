## Test Notes

#### Run the tests
Tests are run using the pytest command in the root of the project. 
Use terminal and run the commands listed below:

- Setup test package:
```
pip install -e ".[tests]"
```

- Execute tests:
```
# Basic
pytest

# Show progression
pytest tests/. -v

# Show branch and cover %
pytest --cov

# Show missing section when cover is not 100%
pytest --cov --cov-report term-missing
```

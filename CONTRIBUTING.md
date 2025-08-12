# Contributing to the batss project

TODO: this needs many more details added.

## Deploying to PyPI
Deploying to the [PyPI website](https://pypi.org/project/batss/) is how users are able to install via `pip install batss`. This is done by a GitHub action in .github/workflows/build_and_publish.yml. It builds binary wheels (so that users do not need a Rust compiler to install) for each major platform and Python version and uploads them to PyPI. This is done automatically whenever there is a new GitHub release. The steps are

1. Bump the version number in pyproject.toml, e.g., change 1.0.0 to 1.0.1 if the last uploaded version was 1.0.0 (or bump minor or major version numbers if appropriate according to [semantic versioning](https://semver.org/)). For the rest of this section assume the version number is 1.0.1.

2. Commit this and other changes to the main branch and push to github.

3. On the [Github page](https://github.com/UC-Davis-molecular-computing/batss), click on Releases-->Create a new release. Title the release `v1.0.1` (or whatever is the version number). This can actually be anything but that is a good title. More importantly, press `Tag: Select tag` and type `v1.0.1`; this must be exactly that, i.e., it must be the lowercase letter `v` followed by the version number in pyproject.toml. Press Enter (i.e., don't just click outside once you've typed; you have to press Enter for it to create the tag). Importantly, there must be no tag already called `v1.0.1`; to double-check, click on Tags at the top of the releases page to see existing tags. If that tag exists, delete it. 

4. Click Publish release.

5. Click on Actions at the top of the page to ensure that the build_and_publish.yml action is running. Because it must compile Rust files for many different platforms (and for each many different Python versions), this takes several minutes. If you are testing out something repeatedly in a short time, it is best to delete all or most of the parts of the Github action that deal with building, since that will eat up limited computation time alloted by Github.

6. If the action successfully completes, go to https://pypi.org/project/batss/ to verify that it is the latest version, 1.0.1 in this case.

7. Recall above that the tag must not already be present. If you need to redo this due to a mistake, not only must you delete the release first, you must also click on Tags at the top of the Releases page and delete the tag. If you do not, then it will not successfully upload to PyPI.
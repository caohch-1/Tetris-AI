# Contributing guide

Here are some ways you can contribute:
- Submitting bug reports and feature requests
- Improving documentation
- Contributing code to fix issues or implement requested features.

Please ensure that all contributions follow the code of conduct laid out in [```CODE_OF_CONDUCT.md```][CODE_OF_CONDUCT]. Questions should be submitted [here](https://github.com/OliverOverend/gym-simplifiedtetris/discussions/new).

## Feature requests

See the Future work section in the ```README.md``` file for ideas. Please create a new GitHub [issue](https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=enhancement&template=FEATURE_REQUEST.md&title=%5BFEATURE%5D%3A) for any significant changes that you wish to make.

Small changes can directly be crafted and submitted to the GitHub repository as a pull request. See the pull request process section below for more details.

## Reporting bugs

**If you find a security vulnerability, do NOT open an issue. Please email ollyoverend10@gmail.com instead.**

Before submitting your issue, please [search the issue archive][Issue tracker] — maybe your question or issue has already been identified or addressed.

If you find a bug in the source code, please [report the bug](https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=bug&template=BUG_REPORT.md&title=%5BBUG%5D%3A) to the GitHub issue tracker. Even better, you can submit a pull request with a fix.

## Improving documentation

If you want to help improve the docs, please create a new issue (or comment on a related existing one) to let others know what you're working on. If you're making a small change (typo, phrasing), don't worry about filing an issue first.

## Contributing code

If you would like to contribute, I'd recommend following [this](https://thenewstack.io/getting-legit-with-git-and-github-your-first-pull-request/) advice. In summary, fork the repo ➡️ clone the copied repo ➡️ create a new branch ➡️ make changes ➡️ merge to master ➡️ create a new pull request.

### Finding an issue

You can find the list of outstanding feature requests and bugs on the [GitHub issue tracker][Issue tracker]. Pick an unassigned issue that you think you can accomplish and add a comment saying that you are trying to resolve the issue.

### Development process

It would be best if you use the master branch for the most stable release.  We work in the 'dev' branch if you want to keep up with the latest changes.  If you are using dev, keep an eye on commits.

### Git commit guidelines

Please follow the format for git commits laid out in [this](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) blog post. You may find [this](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor) commit message editor helpful to generate commit messages.

## Pull request process

Before submitting a pull request, please ensure that you meet these requirements:

- My code follows the same style as the code in this project
- I have formatted my code using [Black](https://github.com/psf/black)
- I have commented my code, particularly in hard-to-understand areas
- I have made corresponding changes to the documentation
- My changes generate no new warnings
- I have added tests that prove my fix is effective or that my feature works
- New and existing unit tests pass locally with my changes
- I have updated the README.md with details of changes (if necessary)
- I have increased the version number in ```setup.py``` to the new version that this pull request would represent. The versioning scheme we use is [SemVer](http://semver.org/).

When you are ready to generate a pull request, you must first push your local topic branch back up to GitHub:

```
git push origin newfeature
```

Once you've committed and pushed all of your changes to GitHub, go to the page for your fork on GitHub, select your development branch, and click the pull request button. Your pull request will automatically track the changes on your development branch and update. If you need to make any adjustments to your pull request, push the updates to your branch.

[CODE_OF_CONDUCT]: https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/CODE_OF_CONDUCT.md
[Getting started]: https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github
[Issue tracker]: https://github.com/OliverOverend/gym-simplifiedtetris/issues

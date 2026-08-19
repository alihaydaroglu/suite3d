@echo off
set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto html
if "%1" == "html" goto html
if "%1" == "clean" goto clean

echo Unknown target: %1
exit /b 1

:html
%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html"
exit /b %ERRORLEVEL%

:clean
if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
exit /b %ERRORLEVEL%

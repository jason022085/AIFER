#!/bin/bash
cd AndroidStudioProject
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
./gradlew lintDebug testDebugUnitTest

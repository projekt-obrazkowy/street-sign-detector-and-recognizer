FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -yq libopencv-dev libopencv4.2-java openjdk-11-jdk

WORKDIR /app
COPY . /app
RUN ./gradlew fatJar --no-daemon

ENTRYPOINT ["java", "-jar", "/app/build/libs/app.jar"]
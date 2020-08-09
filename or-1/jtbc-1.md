---
description: '본 프로젝트에서는 동영상 형식의 뉴스 영상을 음성인식을 통해 추출하고, 그 내용을 요약하는 프로그램을 구현해보고자 한다.'
---

# JTBC 뉴스 요약 봇 프로젝트 \(1\) 소스 코드 준비

※ 본 내용은 YouTube "생각하는 코딩" 채널의 &lt;음성인식 활용 - JTBC 뉴스 요약 봇 프로젝트&gt;를 학습의 목적으로 구현해본 것입니다.

### 사전 준비

1\) ffmpeg 설치 - ETRI OpenAPI를 사용하는데 제약이 있어 사용한다.

* 샘플링 레이트 16kHz
* mono channel만 지원 \(동영상에서 음성 파일과 영상 파일을 구분\)
* 60초 이하 파일만 사용가
* ffmpeg 설치 방법 : [https://ai-creator.tistory.com/78](https://ai-creator.tistory.com/78)

2\) youtube-dl 설

* YouTube 상 뉴스영상을 다운받을 수 있게 하는 프로그램
* youtube-dl 설치 방법 : [https://ai-creator.tistory.com/77](https://ai-creator.tistory.com/77)

3\) pydub 라이브러리 설치

![](../.gitbook/assets/image%20%2836%29.png)

4\) ETRI OpenAPI 사용 신청 : [https://ai-creator.tistory.com/58](https://ai-creator.tistory.com/58)




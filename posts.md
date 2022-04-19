---
layout: posts
title: Posts
permalink: /posts/
---

{% for post in site.posts %}
 {% if post.draft_tag == null %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
 {% endif %}
{% endfor %}

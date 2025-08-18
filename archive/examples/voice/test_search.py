from zhipuai import ZhipuAI
 
client = ZhipuAI(api_key="82baa44acf8c48edbdea0942fc3e33dc.EaKCrV10N8HK6K6j")  # 填写您自己的APIKey
 
response = client.web_search.web_search(
   search_engine="search_std",
   search_query="今天东莞松山湖天气",
   count=1,  # 返回结果的条数，范围1-50，默认10
   search_domain_filter="www.sohu.com",  # 只访问指定域名的内容
   search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
   content_size="medium"  # 控制网页摘要的字数，默认medium
)
print(response.search_result[0].content)
"""
Pytest configuration and fixtures for Semantic Retrieval System API tests
"""

import pytest
import requests
import time
from typing import Dict, Any, List

# API Configuration
API_BASE_URL = "http://localhost:8080"

@pytest.fixture(scope="session")
def api_base_url():
    """Provide API base URL"""
    return API_BASE_URL

@pytest.fixture(scope="session", autouse=True)
def ensure_server_ready(api_base_url):
    """Ensure API server is ready before running tests"""
    print("\n⏳ Waiting for server to be ready...")
    
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get(f"{api_base_url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        if i % 10 == 0:  # Print every 10 seconds
            print(f"   Waiting... ({i+1}/60 seconds)")
    
    pytest.fail("❌ Server is not responding after 60 seconds. Please start the API server first.")

@pytest.fixture
def sample_text():
    """Provide sample text for testing"""
    return """
    Trí tuệ nhân tạo (AI) đang trở thành một trong những công nghệ cốt lõi và có ảnh hưởng sâu rộng nhất trong thế kỷ 21. AI không chỉ thay đổi cách con người tương tác với máy móc mà còn mở ra những cơ hội to lớn trong nhiều lĩnh vực như y tế, giáo dục, tài chính, sản xuất, và đặc biệt là xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP). Trong lĩnh vực này, các mô hình transformer tiên tiến như BERT (Bidirectional Encoder Representations from Transformers) và GPT (Generative Pre-trained Transformer) đã đánh dấu những bước tiến vượt bậc, đưa khả năng hiểu và sinh ngôn ngữ của máy lên một tầm cao mới.

Tuy nhiên, khi áp dụng AI vào các ngôn ngữ khác nhau, đặc biệt là tiếng Việt, các nhà nghiên cứu và kỹ sư phải đối mặt với nhiều thách thức riêng biệt. Tiếng Việt là một ngôn ngữ đơn âm tiết, sử dụng dấu thanh điệu phong phú và có cấu trúc từ vựng đặc biệt, chẳng hạn như nhiều từ ghép không được phân cách bằng khoảng trắng. Điều này gây khó khăn lớn cho các hệ thống xử lý ngôn ngữ truyền thống vốn được thiết kế chủ yếu cho tiếng Anh hoặc các ngôn ngữ có đặc điểm ngữ pháp và cấu trúc khác biệt.

Việt Nam, với hơn 95 triệu dân và là một trong những nền kinh tế phát triển nhanh nhất Đông Nam Á, đang ngày càng quan tâm đến việc phát triển các công nghệ AI phù hợp với ngôn ngữ và văn hóa bản địa. Các doanh nghiệp công nghệ trong nước và quốc tế đã bắt đầu đầu tư mạnh mẽ vào nghiên cứu và phát triển các mô hình AI có thể xử lý tiếng Việt một cách chính xác và hiệu quả. Điều này không chỉ mang lại lợi ích kinh tế mà còn góp phần bảo tồn và phát triển ngôn ngữ Việt trong kỷ nguyên số.

Một trong những thách thức lớn nhất trong việc phát triển AI cho tiếng Việt là việc xây dựng các bộ dữ liệu huấn luyện (training datasets) chất lượng cao. Khác với tiếng Anh có sẵn hàng tỷ trang web và tài liệu số, tiếng Việt có nguồn dữ liệu số hạn chế hơn nhiều. Hơn nữa, việc gán nhãn (labeling) và tiền xử lý dữ liệu tiếng Việt đòi hỏi sự hiểu biết sâu sắc về đặc điểm ngôn ngữ học, bao gồm việc xử lý các dấu thanh điệu, từ đồng âm khác nghĩa, và các biểu thức thành ngữ phong phú.

Các mô hình tokenizer đóng vai trò cực kỳ quan trọng trong xử lý ngôn ngữ tự nhiên. Tokenization là quá trình chia nhỏ văn bản thành các đơn vị nhỏ hơn như từ, cụm từ hoặc ký tự để máy tính có thể xử lý. Đối với tiếng Việt, việc tokenization trở nên phức tạp hơn nhiều so với các ngôn ngữ khác do đặc điểm ngôn ngữ học riêng biệt. Ví dụ, từ "Hồ Chí Minh" có thể được tokenize thành ba token riêng biệt hoặc một token duy nhất tùy thuộc vào ngữ cảnh và mục đích sử dụng.

Trong những năm gần đây, cộng đồng nghiên cứu AI Việt Nam đã có những đóng góp đáng kể cho lĩnh vực NLP tiếng Việt. Các nhóm nghiên cứu từ các trường đại học hàng đầu như Đại học Bách khoa Hà Nội, Đại học Quốc gia Hà Nội, và Đại học Quốc gia TP.HCM đã phát triển nhiều mô hình và công cụ xử lý ngôn ngữ tự nhiên chuyên biệt cho tiếng Việt. Những nghiên cứu này không chỉ có giá trị học thuật mà còn có ứng dụng thực tiễn cao trong các sản phẩm và dịch vụ công nghệ.

PhoBERT, được phát triển bởi nhóm nghiên cứu tại VinAI Research, là một trong những mô hình BERT đầu tiên được huấn luyện đặc biệt cho tiếng Việt. Mô hình này đã đạt được những kết quả ấn tượng trên nhiều tác vụ NLP tiếng Việt khác nhau, từ phân loại văn bản đến nhận dạng thực thể có tên. PhoBERT sử dụng một tokenizer được thiết kế riêng để xử lý đặc điểm của tiếng Việt, bao gồm việc xử lý các dấu thanh điệu và từ ghép một cách chính xác.

Bên cạnh PhoBERT, các mô hình như ViT5 (Vietnamese T5) và BARTpho cũng đã được phát triển để phục vụ các tác vụ sinh văn bản và tóm tắt tiếng Việt. Những mô hình này đặc biệt hữu ích trong việc phát triển các ứng dụng như chatbot tiếng Việt, hệ thống tóm tắt tin tức, và công cụ dịch thuật tự động. Việc có những mô hình chuyên biệt cho tiếng Việt giúp cải thiện đáng kể chất lượng và độ chính xác của các ứng dụng AI trong bối cảnh Việt Nam.

Ngành công nghiệp game và giải trí cũng đang tận dụng các công nghệ AI để tạo ra những trải nghiệm tương tác phong phú hơn bằng tiếng Việt. Các nhà phát triển game đang tích hợp các hệ thống đối thoại AI có thể hiểu và phản hồi bằng tiếng Việt tự nhiên, tạo ra những nhân vật NPC (Non-Player Character) thông minh và sinh động. Điều này không chỉ nâng cao trải nghiệm người chơi mà còn góp phần quảng bá văn hóa và ngôn ngữ Việt Nam đến cộng đồng game thủ quốc tế.

Trong lĩnh vực y tế, các hệ thống AI có khả năng xử lý tiếng Việt đang được ứng dụng để phát triển các trợ lý ảo y tế, hệ thống tư vấn sức khỏe trực tuyến, và công cụ phân tích hồ sơ bệnh án điện tử. Những ứng dụng này đặc biệt quan trọng trong bối cảnh Việt Nam, nơi việc tiếp cận dịch vụ y tế chất lượng còn nhiều hạn chế, đặc biệt ở các vùng nông thôn và miền núi. AI có thể giúp cải thiện khả năng tiếp cận dịch vụ y tế và nâng cao chất lượng chăm sóc sức khỏe cho người dân.

Giáo dục là một lĩnh vực khác đang chứng kiến những ứng dụng tích cực của AI tiếng Việt. Các hệ thống học tập thích ứng (adaptive learning) có khả năng hiểu và phản hồi bằng tiếng Việt đang được phát triển để cá nhân hóa trải nghiệm học tập cho từng học sinh. Những hệ thống này có thể phân tích khả năng học tập, điểm mạnh và điểm yếu của học sinh để đề xuất lộ trình học tập phù hợp, giúp nâng cao hiệu quả giáo dục.

Trong lĩnh vực thương mại điện tử, các chatbot và trợ lý ảo có khả năng xử lý tiếng Việt đang trở thành công cụ không thể thiếu để cải thiện trải nghiệm khách hàng. Những hệ thống này có thể hiểu và phản hồi các câu hỏi của khách hàng bằng tiếng Việt một cách tự nhiên và chính xác, giúp tăng tỷ lệ chuyển đổi và cải thiện mức độ hài lòng của khách hàng. Hơn nữa, chúng có thể hoạt động 24/7, cung cấp hỗ trợ khách hàng liên tục mà không cần can thiệp của con người.

Ngân hàng và tài chính cũng đang tích cực ứng dụng AI để phát triển các dịch vụ tài chính thông minh bằng tiếng Việt. Các hệ thống này có thể phân tích hành vi chi tiêu, đưa ra lời khuyên tài chính cá nhân hóa, và thậm chí là phát hiện các giao dịch bất thường để bảo vệ khách hàng khỏi gian lận. Việc có thể giao tiếp bằng tiếng Việt giúp người dùng cảm thấy thoải mái và tin tưởng hơn khi sử dụng các dịch vụ tài chính số.

Một trong những xu hướng đáng chú ý trong phát triển AI tiếng Việt là việc tích hợp các mô hình đa phương thức (multimodal models) có khả năng xử lý không chỉ văn bản mà còn cả hình ảnh, âm thanh và video. Những mô hình này mở ra khả năng phát triển các ứng dụng phong phú hơn, như hệ thống mô tả hình ảnh bằng tiếng Việt, công cụ tạo phụ đề tự động cho video, và trợ lý ảo có khả năng hiểu và phản hồi qua nhiều kênh giao tiếp khác nhau.

Thách thức về đạo đức và quyền riêng tư cũng là những vấn đề quan trọng cần được xem xét khi phát triển AI tiếng Việt. Việc thu thập và xử lý dữ liệu ngôn ngữ có thể chứa thông tin cá nhân nhạy cảm, do đó cần có những biện pháp bảo vệ quyền riêng tư nghiêm ngặt. Hơn nữa, các mô hình AI cần được phát triển và huấn luyện một cách công bằng, tránh những thiên kiến có thể ảnh hưởng đến các nhóm người dùng khác nhau.

Cộng đồng mã nguồn mở đóng vai trò quan trọng trong việc phát triển AI tiếng Việt. Các dự án mã nguồn mở như Vietnamese NLP Toolkit, UETsegmenter, và VnCoreNLP đã cung cấp những công cụ cơ bản cho việc xử lý ngôn ngữ tự nhiên tiếng Việt. Những dự án này không chỉ giúp các nhà phát triển tiết kiệm thời gian và chi phí mà còn thúc đẩy sự hợp tác và chia sẻ kiến thức trong cộng đồng nghiên cứu.

Về mặt kỹ thuật, việc tối ưu hóa hiệu suất của các mô hình AI tiếng Việt là một thách thức không nhỏ. Do đặc điểm phức tạp của ngôn ngữ, các mô hình AI tiếng Việt thường yêu cầu nhiều tài nguyên tính toán hơn so với các mô hình cho ngôn ngữ khác. Điều này đặt ra nhu cầu phát triển các kỹ thuật tối ưu hóa như model compression, quantization, và distillation để giảm kích thước mô hình và tăng tốc độ inference.

Edge computing và việc triển khai AI trên thiết bị di động cũng là những xu hướng quan trọng trong phát triển AI tiếng Việt. Việc có thể chạy các mô hình AI tiếng Việt trực tiếp trên smartphone hoặc các thiết bị IoT mở ra nhiều ứng dụng thực tiễn, từ trợ lý ảo offline đến các ứng dụng dịch thuật tức thời không cần kết nối internet.

Trong tương lai, chúng ta có thể kỳ vọng thấy những mô hình AI tiếng Việt ngày càng tinh vi và chính xác hơn. Với sự phát triển của các kỹ thuật như few-shot learning và zero-shot learning, các mô hình AI sẽ có thể học và thích ứng với những tác vụ mới chỉ với một lượng dữ liệu huấn luyện tối thiểu. Điều này đặc biệt hữu ích cho tiếng Việt, nơi dữ liệu huấn luyện cho một số tác vụ chuyên biệt vẫn còn hạn chế.

Việc phát triển AI tiếng Việt cũng đòi hỏi sự hợp tác chặt chẽ giữa các bên liên quan, bao gồm các nhà nghiên cứu, doanh nghiệp công nghệ, cơ quan chính phủ, và cộng đồng người dùng. Chỉ thông qua sự hợp tác này, chúng ta mới có thể xây dựng được một hệ sinh thái AI tiếng Việt bền vững và phát triển.

Tóm lại, AI tiếng Việt đang trải qua một giai đoạn phát triển mạnh mẽ với nhiều cơ hội và thách thức. Việc đầu tư vào nghiên cứu và phát triển công nghệ này không chỉ mang lại lợi ích kinh tế mà còn góp phần bảo tồn và phát triển ngôn ngữ Việt trong kỷ nguyên số. Với sự nỗ lực chung của cộng đồng, chúng ta có thể kỳ vọng vào một tương lai nơi AI tiếng Việt sẽ trở thành một phần không thể thiếu trong cuộc sống hàng ngày của người Việt Nam.
"""

@pytest.fixture  
def sample_query():
    """Provide sample query for testing"""
    return "Machine Learning và Deep Learning trong AI là gì?"

@pytest.fixture
def sample_markdown_file():
    """Provide sample markdown content for file upload tests"""
    return """# Artificial Intelligence Overview

## Introduction
Artificial Intelligence (AI) is revolutionizing how we interact with technology.

## Machine Learning
Machine Learning is a subset of AI that enables systems to learn from data.

### Supervised Learning
Uses labeled data to train models.

### Unsupervised Learning  
Finds patterns in unlabeled data.

## Deep Learning
Deep Learning uses neural networks with multiple layers.

### Applications
- Computer Vision
- Natural Language Processing  
- Speech Recognition

## Future of AI
AI will continue to transform industries and society.
"""

@pytest.fixture
def context_chunks(api_base_url, sample_text):
    """Generate context chunks for testing (session scoped for reuse)"""
    response = requests.post(
        f"{api_base_url}/context",
        params={"text": sample_text}
    )
    assert response.status_code == 200
    return response.json()

@pytest.fixture
def query_chunks(api_base_url, sample_query):
    """Generate query chunks for testing (session scoped for reuse)"""
    response = requests.post(
        f"{api_base_url}/query", 
        json={"text": sample_query}
    )
    assert response.status_code == 200
    return response.json()

import token
import tokenize
from trism import TritonModel
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import HTTPException

from src.utils.utils import extract_text_from_file, generate_md5_hash, prepare_input_format, prepare_input_for_tokenizer, prepare_rerank_input, extract_tokenizer_name

# logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_config: Dict[str, Any], tokenizer_config: Dict[str, Any]):
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.models = {}
        self._initialize_models()
        self.context_embedding_length = model_config.get("retrieve_context", {}).get("max_length", 512) # model_max_length of context embedding model

    def _initialize_models(self):
        """Initialize all Triton models"""
        try:
            for model_key, config in self.model_config.items():
                logger.info(f"Initializing {model_key} model...")
                self.models[model_key] = TritonModel(
                    model=config["name"],
                    version=config["version"],
                    url=config["url"],
                    grpc=config["grpc"]
                )
                logger.info(f"Successfully initialized {model_key} model")
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    
    def segment_text(self, text: List) -> List[str]:
        """Use SAT model to segment text into chunks"""
        logger.info(f"Calling SAT model...")

        try:
            tokenizer_name = extract_tokenizer_name(self.tokenizer_config, "sat")
            input_tokenizer_array = prepare_input_for_tokenizer(tokenizer_name) 
            input_text_array = prepare_input_format(text, expand_dims= False)
            input = [input_text_array, input_tokenizer_array]
            result = self.models["sat"].run(data=[input])
            logger.info('this is result',result)
            return result

        except Exception as e:
            # Fallback to simple splitting
            if isinstance(text, list) and len(text) > 0:
                text_str = text[0]
            else:
                text_str = str(text)
            sentences = text_str.split('. ')
            return [s.strip() + '.' for s in sentences if s.strip()]
    
    def get_query_embeddings(self, text: List[str]) -> np.ndarray:
        """Get embeddings for query text"""
        try:
            tokenizer_name = extract_tokenizer_name(self.tokenizer_config, "retrieve_query")
            input_tokenizer_array = prepare_input_for_tokenizer(tokenizer_name)
            input_data = prepare_input_format(text, expand_dims= True)
            result = self.models["retrieve_query"].run(data=[input_data, input_tokenizer_array])
            return result["embedding"]
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")
    
    def get_context_embeddings(self, text: List[str]) -> np.ndarray:
        """Get embeddings for context text"""
        try:
            tokenizer_name = extract_tokenizer_name(self.tokenizer_config, "retrieve_context")
            input_tokenizer_array = prepare_input_for_tokenizer(tokenizer_name)
            input_data = prepare_input_format(text, expand_dims= True)
            result = self.models["retrieve_context"].run(data=[input_data, input_tokenizer_array])
            return result["embedding"]
        except Exception as e:
            logger.error(f"Context embedding failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Context embedding failed: {str(e)}")
    
    def rerank_results(self, query_text: List[str], context_texts: List[str]) -> List[float]:
        """Use rerank model to score query-context pairs"""
        try:
        #     scores = []
            # for ctx_text in context_texts:
            tokenizer_name = extract_tokenizer_name(self.tokenizer_config, "rerank")
            input_tokenizer_array = prepare_input_for_tokenizer(tokenizer_name= tokenizer_name)
            input_data = prepare_rerank_input(query_text, context_texts)

            # Call rerank model with both text inputs
            result = self.models["rerank"].run(data=[input_data, input_tokenizer_array])
            # scores.append(result["logits"])
            return result['logits']
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
        

if __name__ == "__main__":
    import os
    TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:7000")
   
    MODEL_CONFIG = {
    "sat": {
        "name": "sati",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    },
    "retrieve_query": {
        "name": "mbert-retrieve-qry",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    },
    "retrieve_context": {
        "name": "mbert-retrieve-ctx",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False,
        "max_length": 512 
    },
    "rerank": {
        "name": "mbert.rerank",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    }
}
    TOKENIZER_CONFIG = {
    "sat": {
        'name': 'pythera/sat'},
    'retrieve_query': {
        'name': 'pythera/mbert-retrieve-qry-base'},
    'retrieve_context': {
        'name': 'pythera/mbert-retrieve-ctx-base'},
    'rerank': {
        'name': 'pythera/mbert-rerank-base'}
}
    
    model_manager = ModelManager(model_config=MODEL_CONFIG, tokenizer_config=TOKENIZER_CONFIG)
     
#     # test retrival
#     VIETNAMESE_5000_WORDS = ["""
#     Trí tuệ nhân tạo (AI) đang trở thành một trong những công nghệ cốt lõi và có ảnh hưởng sâu rộng nhất trong thế kỷ 21. AI không chỉ thay đổi cách con người tương tác với máy móc mà còn mở ra những cơ hội to lớn trong nhiều lĩnh vực như y tế, giáo dục, tài chính, sản xuất, và đặc biệt là xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP). Trong lĩnh vực này, các mô hình transformer tiên tiến như BERT (Bidirectional Encoder Representations from Transformers) và GPT (Generative Pre-trained Transformer) đã đánh dấu những bước tiến vượt bậc, đưa khả năng hiểu và sinh ngôn ngữ của máy lên một tầm cao mới.

#     Tuy nhiên, khi áp dụng AI vào các ngôn ngữ khác nhau, đặc biệt là tiếng Việt, các nhà nghiên cứu và kỹ sư phải đối mặt với nhiều thách thức riêng biệt. Tiếng Việt là một ngôn ngữ đơn âm tiết, sử dụng dấu thanh điệu phong phú và có cấu trúc từ vựng đặc biệt, chẳng hạn như nhiều từ ghép không được phân cách bằng khoảng trắng. Điều này gây khó khăn lớn cho các hệ thống xử lý ngôn ngữ truyền thống vốn được thiết kế chủ yếu cho tiếng Anh hoặc các ngôn ngữ có đặc điểm ngữ pháp và cấu trúc khác biệt.

#     Việt Nam, với hơn 95 triệu dân và là một trong những nền kinh tế phát triển nhanh nhất Đông Nam Á, đang ngày càng quan tâm đến việc phát triển các công nghệ AI phù hợp với ngôn ngữ và văn hóa bản địa. Các doanh nghiệp công nghệ trong nước và quốc tế đã bắt đầu đầu tư mạnh mẽ vào nghiên cứu và phát triển các mô hình AI có thể xử lý tiếng Việt một cách chính xác và hiệu quả. Điều này không chỉ mang lại lợi ích kinh tế mà còn góp phần bảo tồn và phát triển ngôn ngữ Việt trong kỷ nguyên số.

#     Một trong những thách thức lớn nhất trong việc phát triển AI cho tiếng Việt là việc xây dựng các bộ dữ liệu huấn luyện (training datasets) chất lượng cao. Khác với tiếng Anh có sẵn hàng tỷ trang web và tài liệu số, tiếng Việt có nguồn dữ liệu số hạn chế hơn nhiều. Hơn nữa, việc gán nhãn (labeling) và tiền xử lý dữ liệu tiếng Việt đòi hỏi sự hiểu biết sâu sắc về đặc điểm ngôn ngữ học, bao gồm việc xử lý các dấu thanh điệu, từ đồng âm khác nghĩa, và các biểu thức thành ngữ phong phú.

#     Các mô hình tokenizer đóng vai trò cực kỳ quan trọng trong xử lý ngôn ngữ tự nhiên. Tokenization là quá trình chia nhỏ văn bản thành các đơn vị nhỏ hơn như từ, cụm từ hoặc ký tự để máy tính có thể xử lý. Đối với tiếng Việt, việc tokenization trở nên phức tạp hơn nhiều so với các ngôn ngữ khác do đặc điểm ngôn ngữ học riêng biệt. Ví dụ, từ "Hồ Chí Minh" có thể được tokenize thành ba token riêng biệt hoặc một token duy nhất tùy thuộc vào ngữ cảnh và mục đích sử dụng.

#     Trong những năm gần đây, cộng đồng nghiên cứu AI Việt Nam đã có những đóng góp đáng kể cho lĩnh vực NLP tiếng Việt. Các nhóm nghiên cứu từ các trường đại học hàng đầu như Đại học Bách khoa Hà Nội, Đại học Quốc gia Hà Nội, và Đại học Quốc gia TP.HCM đã phát triển nhiều mô hình và công cụ xử lý ngôn ngữ tự nhiên chuyên biệt cho tiếng Việt. Những nghiên cứu này không chỉ có giá trị học thuật mà còn có ứng dụng thực tiễn cao trong các sản phẩm và dịch vụ công nghệ.

#     PhoBERT, được phát triển bởi nhóm nghiên cứu tại VinAI Research, là một trong những mô hình BERT đầu tiên được huấn luyện đặc biệt cho tiếng Việt. Mô hình này đã đạt được những kết quả ấn tượng trên nhiều tác vụ NLP tiếng Việt khác nhau, từ phân loại văn bản đến nhận dạng thực thể có tên. PhoBERT sử dụng một tokenizer được thiết kế riêng để xử lý đặc điểm của tiếng Việt, bao gồm việc xử lý các dấu thanh điệu và từ ghép một cách chính xác.

#     Bên cạnh PhoBERT, các mô hình như ViT5 (Vietnamese T5) và BARTpho cũng đã được phát triển để phục vụ các tác vụ sinh văn bản và tóm tắt tiếng Việt. Những mô hình này đặc biệt hữu ích trong việc phát triển các ứng dụng như chatbot tiếng Việt, hệ thống tóm tắt tin tức, và công cụ dịch thuật tự động. Việc có những mô hình chuyên biệt cho tiếng Việt giúp cải thiện đáng kể chất lượng và độ chính xác của các ứng dụng AI trong bối cảnh Việt Nam.

#     Ngành công nghiệp game và giải trí cũng đang tận dụng các công nghệ AI để tạo ra những trải nghiệm tương tác phong phú hơn bằng tiếng Việt. Các nhà phát triển game đang tích hợp các hệ thống đối thoại AI có thể hiểu và phản hồi bằng tiếng Việt tự nhiên, tạo ra những nhân vật NPC (Non-Player Character) thông minh và sinh động. Điều này không chỉ nâng cao trải nghiệm người chơi mà còn góp phần quảng bá văn hóa và ngôn ngữ Việt Nam đến cộng đồng game thủ quốc tế.

#     Trong lĩnh vực y tế, các hệ thống AI có khả năng xử lý tiếng Việt đang được ứng dụng để phát triển các trợ lý ảo y tế, hệ thống tư vấn sức khỏe trực tuyến, và công cụ phân tích hồ sơ bệnh án điện tử. Những ứng dụng này đặc biệt quan trọng trong bối cảnh Việt Nam, nơi việc tiếp cận dịch vụ y tế chất lượng còn nhiều hạn chế, đặc biệt ở các vùng nông thôn và miền núi. AI có thể giúp cải thiện khả năng tiếp cận dịch vụ y tế và nâng cao chất lượng chăm sóc sức khỏe cho người dân.

#     Giáo dục là một lĩnh vực khác đang chứng kiến những ứng dụng tích cực của AI tiếng Việt. Các hệ thống học tập thích ứng (adaptive learning) có khả năng hiểu và phản hồi bằng tiếng Việt đang được phát triển để cá nhân hóa trải nghiệm học tập cho từng học sinh. Những hệ thống này có thể phân tích khả năng học tập, điểm mạnh và điểm yếu của học sinh để đề xuất lộ trình học tập phù hợp, giúp nâng cao hiệu quả giáo dục.

#     Trong lĩnh vực thương mại điện tử, các chatbot và trợ lý ảo có khả năng xử lý tiếng Việt đang trở thành công cụ không thể thiếu để cải thiện trải nghiệm khách hàng. Những hệ thống này có thể hiểu và phản hồi các câu hỏi của khách hàng bằng tiếng Việt một cách tự nhiên và chính xác, giúp tăng tỷ lệ chuyển đổi và cải thiện mức độ hài lòng của khách hàng. Hơn nữa, chúng có thể hoạt động 24/7, cung cấp hỗ trợ khách hàng liên tục mà không cần can thiệp của con người.

#     Ngân hàng và tài chính cũng đang tích cực ứng dụng AI để phát triển các dịch vụ tài chính thông minh bằng tiếng Việt. Các hệ thống này có thể phân tích hành vi chi tiêu, đưa ra lời khuyên tài chính cá nhân hóa, và thậm chí là phát hiện các giao dịch bất thường để bảo vệ khách hàng khỏi gian lận. Việc có thể giao tiếp bằng tiếng Việt giúp người dùng cảm thấy thoải mái và tin tưởng hơn khi sử dụng các dịch vụ tài chính số.

#     Một trong những xu hướng đáng chú ý trong phát triển AI tiếng Việt là việc tích hợp các mô hình đa phương thức (multimodal models) có khả năng xử lý không chỉ văn bản mà còn cả hình ảnh, âm thanh và video. Những mô hình này mở ra khả năng phát triển các ứng dụng phong phú hơn, như hệ thống mô tả hình ảnh bằng tiếng Việt, công cụ tạo phụ đề tự động cho video, và trợ lý ảo có khả năng hiểu và phản hồi qua nhiều kênh giao tiếp khác nhau.

#     Thách thức về đạo đức và quyền riêng tư cũng là những vấn đề quan trọng cần được xem xét khi phát triển AI tiếng Việt. Việc thu thập và xử lý dữ liệu ngôn ngữ có thể chứa thông tin cá nhân nhạy cảm, do đó cần có những biện pháp bảo vệ quyền riêng tư nghiêm ngặt. Hơn nữa, các mô hình AI cần được phát triển và huấn luyện một cách công bằng, tránh những thiên kiến có thể ảnh hưởng đến các nhóm người dùng khác nhau.

#     Cộng đồng mã nguồn mở đóng vai trò quan trọng trong việc phát triển AI tiếng Việt. Các dự án mã nguồn mở như Vietnamese NLP Toolkit, UETsegmenter, và VnCoreNLP đã cung cấp những công cụ cơ bản cho việc xử lý ngôn ngữ tự nhiên tiếng Việt. Những dự án này không chỉ giúp các nhà phát triển tiết kiệm thời gian và chi phí mà còn thúc đẩy sự hợp tác và chia sẻ kiến thức trong cộng đồng nghiên cứu.

#     Về mặt kỹ thuật, việc tối ưu hóa hiệu suất của các mô hình AI tiếng Việt là một thách thức không nhỏ. Do đặc điểm phức tạp của ngôn ngữ, các mô hình AI tiếng Việt thường yêu cầu nhiều tài nguyên tính toán hơn so với các mô hình cho ngôn ngữ khác. Điều này đặt ra nhu cầu phát triển các kỹ thuật tối ưu hóa như model compression, quantization, và distillation để giảm kích thước mô hình và tăng tốc độ inference.

#     Edge computing và việc triển khai AI trên thiết bị di động cũng là những xu hướng quan trọng trong phát triển AI tiếng Việt. Việc có thể chạy các mô hình AI tiếng Việt trực tiếp trên smartphone hoặc các thiết bị IoT mở ra nhiều ứng dụng thực tiễn, từ trợ lý ảo offline đến các ứng dụng dịch thuật tức thời không cần kết nối internet.

#     Trong tương lai, chúng ta có thể kỳ vọng thấy những mô hình AI tiếng Việt ngày càng tinh vi và chính xác hơn. Với sự phát triển của các kỹ thuật như few-shot learning và zero-shot learning, các mô hình AI sẽ có thể học và thích ứng với những tác vụ mới chỉ với một lượng dữ liệu huấn luyện tối thiểu. Điều này đặc biệt hữu ích cho tiếng Việt, nơi dữ liệu huấn luyện cho một số tác vụ chuyên biệt vẫn còn hạn chế.

#     Việc phát triển AI tiếng Việt cũng đòi hỏi sự hợp tác chặt chẽ giữa các bên liên quan, bao gồm các nhà nghiên cứu, doanh nghiệp công nghệ, cơ quan chính phủ, và cộng đồng người dùng. Chỉ thông qua sự hợp tác này, chúng ta mới có thể xây dựng được một hệ sinh thái AI tiếng Việt bền vững và phát triển.

#     Tóm lại, AI tiếng Việt đang trải qua một giai đoạn phát triển mạnh mẽ với nhiều cơ hội và thách thức. Việc đầu tư vào nghiên cứu và phát triển công nghệ này không chỉ mang lại lợi ích kinh tế mà còn góp phần bảo tồn và phát triển ngôn ngữ Việt trong kỷ nguyên số. Với sự nỗ lực chung của cộng đồng, chúng ta có thể kỳ vọng vào một tương lai nơi AI tiếng Việt sẽ trở thành một phần không thể thiếu trong cuộc sống hàng ngày của người Việt Nam.
#     """]



#     # tokenizer_name = "pythera/mbert-retrieve-qry-base"
#     result = model_manager.segment_text(VIETNAMESE_5000_WORDS)
#     print((len(result)))

    # test rerank    
    query_texts = ["Đâu là thủ đô của Việt Nam?"]
    context_texts = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Pháp là quốc gia tại Châu Âu.",
    "Pháp có tháp nghiêng."
] 
    tokenizer_name = "pythera/mbert-rerank-base"
    result = model_manager.rerank_results(query_texts, context_texts)
    print((result))

#     # test rerank embedding 
#     context_texts = [
#     "Paris is the capital of France.",
#     "France is a country in Europe.",
#     "The Eiffel Tower is in Paris."
# ] 
    # tokenizer_name = "pythera/mbert-retrieve-ctx-base"
    # result = model_manager.rerank_results(query_texts, context_texts, tokenizer_name)
    # print(np.array(result))


    # test sat moudle
    # text = 'Tôi là Nguyễn Nhật Trường và hôm nay là một ngày vui , tôi cảm thấy khá là hạnh phúc'
    # tokenizer_name = 'pythera/sat'
    # result = model_manager.segment_text(text= text, tokenizer_name= tokenizer_name)
    # print(np.array(result))
    